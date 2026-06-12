"""
Cultural Gravity Precomputation
================================
One-time GPU pass to build lookup tables mapping price / volume / market-cap
values to their cosine similarity against cultural concept clusters.

The embedding model has been trained on internet text, so it already "knows"
the cultural associations of numbers — "six million" lives near Holocaust
concepts, "four twenty" lives near cannabis/meme culture, etc.

We exploit that geometry.  No quant compliance desk would sign off on this.

Run once:
    python precompute_cultural_gravity.py

Produces:
    Data/cultural_gravity_lookup.pkl   (~5-15 MB)
"""

import sys, os, pickle
import numpy as np
import torch
from sentence_transformers import SentenceTransformer

sys.stdout.reconfigure(encoding='utf-8')

try:
    from num2words import num2words as _n2w
    _HAS_N2W = True
except ImportError:
    _HAS_N2W = False

_CULTURAL_ALIASES: dict = {
    42:    ["forty two"],
    69:    ["sixty nine"],
    88:    ["eighty eight"],
    108:   ["one hundred eight"],
    111:   ["triple one", "one eleven"],
    222:   ["triple two", "two twenty two"],
    333:   ["triple three", "three thirty three"],
    404:   ["four oh four"],
    420:   ["four twenty"],
    444:   ["triple four", "four forty four"],
    555:   ["triple five", "five fifty five"],
    666:   ["six sixty six", "six six six"],
    777:   ["seven seventy seven", "triple seven"],
    888:   ["eight eighty eight", "triple eight"],
    911:   ["nine eleven"],
    999:   ["triple nine", "nine ninety nine"],
    1337:  ["thirteen thirty seven"],
    1488:  ["fourteen eighty eight", "fourteen words"],
    4.20:  ["four twenty"],
    6.66:  ["six sixty six"],
    6.69:  ["six sixty nine"],
    13.37: ["thirteen thirty seven"],
    66.6:  ["sixty six point six"],
}

DEVICE  = 'cuda:0'          # primary 3070
MODEL   = 'all-mpnet-base-v2'   # 109M params, 768-dim, better cultural coverage than MiniLM
OUT     = 'Data/cultural_gravity_lookup.pkl'

# ── Concept anchor clusters ────────────────────────────────────────────────────
# Each cluster defines a "direction" in embedding space.
# We measure cosine similarity from a number-phrase to each centroid.
# The clusters are chosen to capture the *specific* information gaps that
# no institutional research team is allowed to exploit systematically.

ANCHORS = {

    # Numbers that make people hesitate — associated with atrocity, forbidden
    # history, or taboo.  The thesis: these create a fractional friction in
    # order flow that is too uncomfortable to be systematically arbed.
    'dark_taboo': [
        'six million victims genocide death toll',
        'concentration camp six million dead',
        'holocaust death toll total victims',
        'war crime civilian casualties mass murder',
        'six six six number of the beast antichrist',
        'satanic ritual occult cursed number omen',
        'death count fatalities atrocity mass grave',
        'forbidden cursed ominous number evil',
        'three hundred thousand casualties massacre',
        'two hundred seventy thousand dead genocide',
    ],

    # Numbers that attract retail eyeballs and cause social posting.
    # When a price or volume hits a meme number, retail posts screenshots.
    # That attention spike has a measurable, if brief, impact on flow.
    'meme_viral': [
        'sixty nine sexual innuendo internet joke',
        'four hundred twenty cannabis weed marijuana culture',
        'over nine thousand power level dragon ball meme',
        'thirteen thirty seven elite hacker leet speak',
        'forty two meaning of life the universe everything',
        'stonks meme stock reddit wallstreetbets to the moon',
        'yolo gambling degen ape gamestop short squeeze',
        'one million subscribers viral famous internet celebrity',
        'doge much wow very moon cryptocurrency meme coin',
        'based cringe sigma alpha chad internet culture',
    ],

    # Numbers with sacred, mystical, or religious significance.
    # These show up in numerology, religious texts, and ritual — different
    # from dark_taboo but also create non-rational human responses.
    'sacred_mystical': [
        'seven lucky divine holy sacred number',
        'seven seven seven jackpot slot machine fortune',
        'one hundred eight sacred Buddhist number meditation',
        'thirteen unlucky superstition bad luck',
        'three trinity holy father son spirit',
        'perfect number mathematics integer divine ratio',
        'fibonacci golden ratio natural mathematics',
        'pi three point one four mathematics constant universe',
        'one thousand and one Arabian nights mystical legend',
        'forty days forty nights biblical scripture covenant',
    ],

    # Numbers at psychological round-number anchors.
    # This is documented behavioral finance — price clustering at round numbers.
    # Anchored here to have a "normal" baseline to compare against.
    'round_magnetic': [
        'one hundred milestone century achievement',
        'one thousand barrier threshold breakthrough',
        'all time high breakout record new high resistance',
        'psychological price level support resistance key level',
        'fifty percent halfway retracement midpoint',
        'round number order clustering price magnet',
        'one billion market capitalization large cap',
        'trillion dollar valuation mega cap blue chip',
    ],

}

# ── Number → phrase representations ───────────────────────────────────────────

def price_phrases(p: float) -> list[str]:
    """Multiple string representations of a stock price."""
    out = [
        f"{p:.2f}",
        f"${p:.2f} stock price",
        f"{p:.0f} dollars per share",
    ]
    if p >= 1000:
        out.append(f"{p/1000:.2f} thousand dollar stock")
    cents = round(p % 1, 2)
    whole = int(p)
    out.append(f"{whole} dollars and {int(cents*100):02d} cents")

    if _HAS_N2W:
        try:
            word = _n2w(int(p))
            out.append(word)
            out.append(f"{word} dollars")
        except Exception:
            pass

    aliases = _CULTURAL_ALIASES.get(round(p, 2)) or _CULTURAL_ALIASES.get(int(p))
    if aliases:
        out.extend(aliases)

    return out


def volume_phrases(v: float) -> list[str]:
    """Multiple string representations of a share volume."""
    out = []
    if v >= 1e9:
        out.append(f"{v/1e9:.2f} billion shares traded")
        out.append(f"{v/1e9:.2f} billion volume")
    elif v >= 1e6:
        out.append(f"{v/1e6:.2f} million shares traded")
        out.append(f"{v/1e6:.2f} million volume")
    elif v >= 1e3:
        out.append(f"{v/1e3:.1f} thousand shares")
    out.append(f"{int(v):,} shares")
    return out if out else [f"{int(v)} shares"]


def mktcap_phrases(mc: float) -> list[str]:
    """Multiple string representations of a market capitalisation."""
    out = []
    if mc >= 1e12:
        out.append(f"{mc/1e12:.2f} trillion dollar company")
        out.append(f"{mc/1e12:.2f} trillion market cap")
    elif mc >= 1e9:
        out.append(f"{mc/1e9:.2f} billion dollar company")
        out.append(f"{mc/1e9:.2f} billion market cap")
    elif mc >= 1e6:
        out.append(f"{mc/1e6:.2f} million dollar company")
    out.append(f"${mc:,.0f} market capitalisation")
    return out


# ── Grid of values to precompute ───────────────────────────────────────────────

def make_price_grid() -> list[float]:
    grid = list(range(1, 2001))                          # $1 – $2000 integer steps
    # Add fractional loaded prices
    extra = [
        0.69, 4.20, 6.66, 6.69, 13.37, 42.0, 66.6, 69.0,
        88.0, 108.0, 133.7, 420.0, 666.0, 777.0, 1337.0,
    ]
    return sorted(set([float(g) for g in grid] + extra))


def make_volume_grid() -> list[float]:
    # Log-spaced from 10k to 5B shares
    return sorted(set([
        *np.arange(10_000,    100_000,   10_000).tolist(),
        *np.arange(100_000,   1_000_000, 50_000).tolist(),
        *np.arange(1_000_000, 10_000_000, 500_000).tolist(),
        *np.arange(10_000_000, 100_000_000, 5_000_000).tolist(),
        *np.arange(100_000_000, 1_000_000_000, 50_000_000).tolist(),
        # Culturally loaded volumes (rounded)
        271_000, 6_000_000, 6_660_000, 666_000,
        420_000, 69_000, 1_337_000,
    ]))


def make_mktcap_grid() -> list[float]:
    # From $10M to $3T log-spaced
    return sorted(set([
        *np.logspace(7, 12, 200).tolist(),               # $10M to $1T
        6_000_000_000, 6_660_000_000, 271_000_000,
    ]))


# ── Batch encode + cosine similarity ──────────────────────────────────────────

def build_table(
    values: list[float],
    phrase_fn,
    anchor_centroids: dict[str, np.ndarray],
    model: SentenceTransformer,
    batch_size: int = 512,
    label: str = '',
) -> dict[float, dict[str, float]]:
    """For each value, embed its phrase representations and score against anchors."""

    # Flatten all phrases, remembering which value each belongs to
    all_phrases, offsets, lengths = [], [], []
    for v in values:
        phrases = phrase_fn(v)
        offsets.append(len(all_phrases))
        lengths.append(len(phrases))
        all_phrases.extend(phrases)

    print(f'  {label}: encoding {len(all_phrases)} phrases for {len(values)} values ...')
    embs = model.encode(
        all_phrases,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,   # L2-normalised → dot product = cosine sim
    )

    # Pre-normalise anchor centroids
    normed_anchors = {}
    for name, c in anchor_centroids.items():
        n = np.linalg.norm(c)
        normed_anchors[name] = c / n if n > 1e-10 else c

    table = {}
    for i, v in enumerate(values):
        chunk = embs[offsets[i]: offsets[i] + lengths[i]]
        avg   = chunk.mean(axis=0)
        norm  = np.linalg.norm(avg)
        avg   = avg / norm if norm > 1e-10 else avg
        scores = {name: float(np.dot(avg, c)) for name, c in normed_anchors.items()}
        table[v] = scores

    return table


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print(f'Loading model: {MODEL} on {DEVICE}')
    model = SentenceTransformer(MODEL, device=DEVICE)
    print(f'  Embedding dim: {model.get_sentence_embedding_dimension()}')

    # Compute anchor centroids
    print('\nComputing anchor centroids ...')
    anchor_centroids: dict[str, np.ndarray] = {}
    for name, phrases in ANCHORS.items():
        embs = model.encode(phrases, convert_to_numpy=True, normalize_embeddings=True)
        centroid = embs.mean(axis=0)
        anchor_centroids[name] = centroid
        print(f'  {name}: {len(phrases)} phrases → centroid norm={np.linalg.norm(centroid):.4f}')

    # Build lookup tables
    price_grid  = make_price_grid()
    vol_grid    = make_volume_grid()
    mktcap_grid = make_mktcap_grid()

    print(f'\nGrids: {len(price_grid)} prices | {len(vol_grid)} volumes | {len(mktcap_grid)} mktcaps')

    price_table  = build_table(price_grid,  price_phrases,  anchor_centroids, model, label='price')
    vol_table    = build_table(vol_grid,    volume_phrases, anchor_centroids, model, label='volume')
    mktcap_table = build_table(mktcap_grid, mktcap_phrases, anchor_centroids, model, label='mktcap')

    output = {
        'model':         MODEL,
        'anchor_names':  list(ANCHORS.keys()),
        'anchors':       ANCHORS,
        'price_table':   price_table,
        'vol_table':     vol_table,
        'mktcap_table':  mktcap_table,
        'price_grid':    price_grid,
        'vol_grid':      vol_grid,
        'mktcap_grid':   mktcap_grid,
    }

    os.makedirs('Data', exist_ok=True)
    with open(OUT, 'wb') as f:
        pickle.dump(output, f, protocol=5)
    print(f'\nSaved → {OUT}  ({os.path.getsize(OUT)/1024:.0f} KB)')

    # Quick sanity check — print top-5 most dark_taboo prices
    print('\nTop-10 dark_taboo prices:')
    ranked = sorted(price_table.items(), key=lambda kv: kv[1]['dark_taboo'], reverse=True)[:10]
    for p, s in ranked:
        print(f'  ${p:<10.2f}  dark={s["dark_taboo"]:+.4f}  meme={s["meme_viral"]:+.4f}  sacred={s["sacred_mystical"]:+.4f}')

    print('\nTop-10 meme_viral prices:')
    ranked = sorted(price_table.items(), key=lambda kv: kv[1]['meme_viral'], reverse=True)[:10]
    for p, s in ranked:
        print(f'  ${p:<10.2f}  dark={s["dark_taboo"]:+.4f}  meme={s["meme_viral"]:+.4f}  sacred={s["sacred_mystical"]:+.4f}')

    print('\nTop-10 dark_taboo volumes:')
    ranked = sorted(vol_table.items(), key=lambda kv: kv[1]['dark_taboo'], reverse=True)[:10]
    for v, s in ranked:
        print(f'  {v/1e6:>10.3f}M  dark={s["dark_taboo"]:+.4f}  meme={s["meme_viral"]:+.4f}')


if __name__ == '__main__':
    main()
