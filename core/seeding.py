"""Run seeding: one recorded run seed fanned into named, independent streams.

Every stochastic component of a run derives its own stream from ``(run_seed, labels)``
by hashing — never by splitting or copying a shared generator. Two properties follow,
and they are the whole point:

  * **Independence.** Consuming one purpose's stream cannot perturb another's sequence,
    so adding a draw to (say) move selection does not silently reshuffle augmentation.
  * **Re-derivability.** Any stream is recomputable from ``(run_seed, labels)`` alone, so
    parallel actors decorrelate by label and crash-resume needs no persisted RNG state —
    M3 deepens the keying to durable coordinates (``("actor", id, "game", i, purpose)``,
    ``("learner", step, "replay-sampling")``) without reshaping this API.

The single-process M2.5 label set is the purpose constants below: net init, Dirichlet
(D7), ∝N move selection (D10), MCTS tie-breaks, symmetry-augmentation ``g`` (D9), and
replay-window sampling. Pure stdlib, no torch/NumPy: net init consumes the derived int
directly via ``torch.manual_seed``.
"""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass

# Domain separator: seeds derived here can never collide with another use of sha256 over
# similar-looking bytes, and bumping the version deliberately re-keys every stream.
_DOMAIN = b"alpha-games/seeding/v1"
_SEED_BYTES = 8

Label = str | int

# The single-process purpose names (M3 adds the multi-actor label families).
PURPOSE_NET_INIT = "net-init"
PURPOSE_DIRICHLET = "dirichlet"
PURPOSE_MOVE_SELECTION = "move-selection"
PURPOSE_TIE_BREAK = "tie-break"
PURPOSE_AUGMENTATION = "augmentation"
PURPOSE_WINDOW_SAMPLING = "replay-sampling"


def _encode(part: Label) -> bytes:
    """Encode one label part as unambiguous, self-delimiting bytes.

    Type-tagged and length-prefixed, so no two distinct label tuples can serialize to the
    same byte string: ``("a", 1)``, ``("a1",)`` and ``(1, "a")`` are all different.

    Args:
        part: A label part — ``str`` or ``int``.

    Returns:
        The encoded bytes, ``b"<tag>:<len>:<raw>"``.

    Raises:
        TypeError: If ``part`` is neither ``str`` nor ``int`` (``bool`` is rejected: it is
            an ``int`` subclass and reads as a flag, never as a stable label).
    """
    if isinstance(part, bool):
        raise TypeError("seed labels must be str or int, not bool")
    if isinstance(part, int):
        tag, raw = b"i", str(part).encode("ascii")
    elif isinstance(part, str):
        tag, raw = b"s", part.encode("utf-8")
    else:
        raise TypeError(f"seed labels must be str or int; got {type(part).__name__}")
    return b"%s:%d:%s" % (tag, len(raw), raw)


def derive_seed(run_seed: int, *labels: Label) -> int:
    """Derive a component seed from the run seed and a label tuple.

    sha256 over a canonical serialization of ``(run_seed, labels)``, first 8 bytes as a
    big-endian unsigned int. Deterministic across processes and runs: the same inputs
    always give the same seed, which is what makes actor decorrelation and crash-resume
    a matter of recomputation rather than bookkeeping.

    Args:
        run_seed: The run's recorded root seed.
        *labels: Stable label parts, e.g. ``("actor", 3, "dirichlet")`` or ``("net-init",)``.

    Returns:
        A seed in ``[0, 2**64)``, suitable for ``random.Random`` or ``torch.manual_seed``.

    Raises:
        TypeError: If ``run_seed`` is not an ``int`` or any label part is not ``str``/``int``.
    """
    if isinstance(run_seed, bool) or not isinstance(run_seed, int):
        raise TypeError(f"run_seed must be an int; got {type(run_seed).__name__}")
    payload = _DOMAIN + _encode(run_seed) + b"".join(_encode(part) for part in labels)
    return int.from_bytes(hashlib.sha256(payload).digest()[:_SEED_BYTES], "big")


def component_rng(run_seed: int, *labels: Label) -> random.Random:
    """Return the ``random.Random`` for one named stream.

    Args:
        run_seed: The run's recorded root seed.
        *labels: The stream's label parts (see :func:`derive_seed`).

    Returns:
        A fresh generator seeded with ``derive_seed(run_seed, *labels)``.

    Raises:
        TypeError: If ``run_seed`` or a label part has an unsupported type.
    """
    return random.Random(derive_seed(run_seed, *labels))


def net_init_seed(run_seed: int) -> int:
    """Return the network-initialization seed (fed straight to ``torch.manual_seed``).

    Args:
        run_seed: The run's recorded root seed.

    Returns:
        The derived seed for label ``("net-init",)``.
    """
    return derive_seed(run_seed, PURPOSE_NET_INIT)


@dataclass(frozen=True)
class GameRNGs:
    """The per-purpose generators one self-play game runs on.

    One stream per purpose — never one rng threading every stochastic choice — so a change
    in how many Dirichlet draws a search makes cannot shift the move-selection sequence.
    ``play_game`` takes this bundle; M3 deepens the keying via ``prefix`` (its
    ``("actor", actor_id, ...)`` family) without changing the bundle's shape.

    Attributes:
        dirichlet: D7 root-noise stream, handed to ``MCTS(root_noise=...)``.
        move_selection: D10 ∝N sampling stream.
        tie_break: MCTS tie-break stream (kept apart from move selection so a tie-break
            draw never shifts the played-move sequence).
    """

    dirichlet: random.Random
    move_selection: random.Random
    tie_break: random.Random

    @classmethod
    def for_game(
        cls, run_seed: int, game_index: int, *, prefix: tuple[Label, ...] = ()
    ) -> GameRNGs:
        """Build the bundle for one game from durable coordinates.

        Labels are ``(*prefix, "game", game_index, purpose)``. ``game_index`` must be a
        durable counter (M3: the actor's persisted next-game index), never an in-process
        position — that is what makes a game reproducible on its own after a restart.

        Args:
            run_seed: The run's recorded root seed.
            game_index: The game's durable index within the run.
            prefix: Extra leading label parts, e.g. ``("actor", 3)`` at M3. Empty in the
                single-process M2.5 loop.

        Returns:
            A frozen bundle of independently derived generators.

        Raises:
            TypeError: If ``run_seed``, ``game_index`` or a prefix part has an unsupported type.
        """
        labels = (*prefix, "game", game_index)
        return cls(
            dirichlet=component_rng(run_seed, *labels, PURPOSE_DIRICHLET),
            move_selection=component_rng(run_seed, *labels, PURPOSE_MOVE_SELECTION),
            tie_break=component_rng(run_seed, *labels, PURPOSE_TIE_BREAK),
        )


@dataclass(frozen=True)
class LearnerRNGs:
    """The per-purpose generators one learner step runs on.

    Keyed per step rather than held across the run, so the streams a step draws on are
    recomputable from the checkpointed step counter alone (M3's crash-resume rule).

    Attributes:
        augmentation: D9 symmetry-``g`` choice per sampled position.
        window_sampling: Replay-window batch sampling.
    """

    augmentation: random.Random
    window_sampling: random.Random

    @classmethod
    def for_step(
        cls, run_seed: int, step: int, *, prefix: tuple[Label, ...] = ("learner",)
    ) -> LearnerRNGs:
        """Build the bundle for one learner step.

        Labels are ``(*prefix, step, purpose)`` — matching M3's
        ``("learner", learner_step, "replay-sampling")``.

        Args:
            run_seed: The run's recorded root seed.
            step: The learner step number (durable: it is checkpointed).
            prefix: Leading label parts; defaults to ``("learner",)``.

        Returns:
            A frozen bundle of independently derived generators.

        Raises:
            TypeError: If ``run_seed``, ``step`` or a prefix part has an unsupported type.
        """
        labels = (*prefix, step)
        return cls(
            augmentation=component_rng(run_seed, *labels, PURPOSE_AUGMENTATION),
            window_sampling=component_rng(run_seed, *labels, PURPOSE_WINDOW_SAMPLING),
        )
