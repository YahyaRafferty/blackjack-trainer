"""
bankroll_bot_sim.py

Usage
-----
    python bankroll_bot_sim.py                     # basic strategy, default rules
    python bankroll_bot_sim.py --hands 5000         # simulate 5,000 hands per bot
    python bankroll_bot_sim.py --bots 1000          # simulate 1,000 bots per batch
    python bankroll_bot_sim.py --decks 6            # 4, 6, or 8 decks
    python bankroll_bot_sim.py --dealer-rule s17    # dealer stands soft 17
    python bankroll_bot_sim.py --strategy basic     # basic strategy play
    python bankroll_bot_sim.py --no-plot            # skip charts, numbers only

Inputs
------
    --dealer-rule   s17 or h17               (default: s17)
    --strategy      basic, hilo, or omega2   (default: basic)
    --decks         any positive integer     (default: 6)
    --bots          any positive integer     (default: 1000)
    --hands         any positive integer     (default: 5000)
"""

import argparse
import csv
import math
import os
import random
from collections import deque

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

NUM_BOTS = 1000
NUM_HANDS = 5000
BASE_SEED = 42
ZOOM_HANDS = 300
DECOMPOSITION_BOT = 0
OUTPUT_DIR = 'bankroll_sim_output'

STARTING_BANKROLL = 10_000.0


NUM_DECKS = 6
DEALER_HITS_SOFT_17 = False        # False = stand on soft 17 (S17), True = hit (H17)
MAX_SPLIT_HANDS = 2                # 2 = one split allowed, no resplitting
RESHUFFLE_CUTOFF = 15              # reshuffle once fewer than this many cards remain

STRATEGY_MODE = 'basic'            # 'basic', 'hilo', or 'omega2'
BET_SPREAD_MAX = 12                # betting units cap when count-based betting is active
MIN_DECKS_REMAINING = 0.5          # floor on decks-remaining when computing true count,
                                    # so true count doesn't blow up near the shoe's end

# 2-9 are face value, 10 represents 10/J/Q/K, 11 represents an Ace (soft
# value; hand_value() converts a soft Ace down to 1 automatically when needed)
RANK_VALUES = [2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10, 11]

# Per-card count tags for each system (indexed by the same card values used
# above). True count = running count / decks remaining; a positive true
# count means the remaining shoe is richer in tens/aces than average.
COUNT_SYSTEMS = {
    'hilo':   {2: 1, 3: 1, 4: 1, 5: 1, 6: 1, 7: 0, 8: 0, 9: 0, 10: -1, 11: -1},
    'omega2': {2: 1, 3: 1, 4: 2, 5: 2, 6: 2, 7: 1, 8: 0, 9: -1, 10: -2, 11: 0},
}

# True-count thresholds at or above which the side bet / play deviates from
# basic strategy. This is a representative teaching subset of the well-known
# Hi-Lo "Illustrious 18" deviations, reused for Omega II as a simplifying
# assumption (real Omega II index numbers differ somewhat and, for a few
# plays, would need a separate Ace side-count for full accuracy).
INSURANCE_INDEX = {'hilo': 3, 'omega2': 3}

INDEX_PLAYS = {
    ('hard', 16, 10): (0, 'S'),
    ('hard', 16, 9):  (5, 'S'),
    ('hard', 15, 10): (4, 'S'),
    ('hard', 12, 3):  (2, 'S'),
    ('hard', 12, 2):  (3, 'S'),
    ('hard', 10, 10): (4, 'Dh'),
    ('hard', 10, 11): (4, 'Dh'),
    ('hard', 11, 11): (1, 'Dh'),
    ('hard', 9, 2):   (1, 'Dh'),
    ('hard', 9, 7):   (3, 'Dh'),
    ('pair', 10, 5):  (5, 'P'),
    ('pair', 10, 6):  (4, 'P'),
}

# Action codes: H = hit, S = stand, Dh = double else hit,
# Ds = double else stand, P = split, Rh = surrender else hit
HARD_TOTALS = {
    8:  {2: 'H', 3: 'H', 4: 'H', 5: 'H', 6: 'H', 7: 'H', 8: 'H', 9: 'H', 10: 'H', 11: 'H'},
    9:  {2: 'H', 3: 'Dh', 4: 'Dh', 5: 'Dh', 6: 'Dh', 7: 'H', 8: 'H', 9: 'H', 10: 'H', 11: 'H'},
    10: {2: 'Dh', 3: 'Dh', 4: 'Dh', 5: 'Dh', 6: 'Dh', 7: 'Dh', 8: 'Dh', 9: 'Dh', 10: 'H', 11: 'H'},
    11: {2: 'Dh', 3: 'Dh', 4: 'Dh', 5: 'Dh', 6: 'Dh', 7: 'Dh', 8: 'Dh', 9: 'Dh', 10: 'Dh', 11: 'H'},
    12: {2: 'H', 3: 'H', 4: 'S', 5: 'S', 6: 'S', 7: 'H', 8: 'H', 9: 'H', 10: 'H', 11: 'H'},
    13: {2: 'S', 3: 'S', 4: 'S', 5: 'S', 6: 'S', 7: 'H', 8: 'H', 9: 'H', 10: 'H', 11: 'H'},
    14: {2: 'S', 3: 'S', 4: 'S', 5: 'S', 6: 'S', 7: 'H', 8: 'H', 9: 'H', 10: 'H', 11: 'H'},
    15: {2: 'S', 3: 'S', 4: 'S', 5: 'S', 6: 'S', 7: 'H', 8: 'H', 9: 'H', 10: 'Rh', 11: 'H'},
    16: {2: 'S', 3: 'S', 4: 'S', 5: 'S', 6: 'S', 7: 'H', 8: 'H', 9: 'Rh', 10: 'Rh', 11: 'Rh'},
    # 17+ always stands
}

SOFT_TOTALS = {
    13: {2: 'H', 3: 'H', 4: 'H', 5: 'Dh', 6: 'Dh', 7: 'H', 8: 'H', 9: 'H', 10: 'H', 11: 'H'},
    14: {2: 'H', 3: 'H', 4: 'H', 5: 'Dh', 6: 'Dh', 7: 'H', 8: 'H', 9: 'H', 10: 'H', 11: 'H'},
    15: {2: 'H', 3: 'H', 4: 'Dh', 5: 'Dh', 6: 'Dh', 7: 'H', 8: 'H', 9: 'H', 10: 'H', 11: 'H'},
    16: {2: 'H', 3: 'H', 4: 'Dh', 5: 'Dh', 6: 'Dh', 7: 'H', 8: 'H', 9: 'H', 10: 'H', 11: 'H'},
    17: {2: 'H', 3: 'Dh', 4: 'Dh', 5: 'Dh', 6: 'Dh', 7: 'H', 8: 'H', 9: 'H', 10: 'H', 11: 'H'},
    18: {2: 'Ds', 3: 'Ds', 4: 'Ds', 5: 'Ds', 6: 'Ds', 7: 'S', 8: 'S', 9: 'H', 10: 'H', 11: 'H'},
    19: {2: 'S', 3: 'S', 4: 'S', 5: 'S', 6: 'Ds', 7: 'S', 8: 'S', 9: 'S', 10: 'S', 11: 'S'},
    # 20-21 always stands
}

# Keyed by pair rank value (2-11, where 10 covers all ten-value cards)
PAIR_TOTALS = {
    11: {2: 'P', 3: 'P', 4: 'P', 5: 'P', 6: 'P', 7: 'P', 8: 'P', 9: 'P', 10: 'P', 11: 'P'},
    9:  {2: 'P', 3: 'P', 4: 'P', 5: 'P', 6: 'P', 7: 'S', 8: 'P', 9: 'P', 10: 'S', 11: 'S'},
    8:  {2: 'P', 3: 'P', 4: 'P', 5: 'P', 6: 'P', 7: 'P', 8: 'P', 9: 'P', 10: 'P', 11: 'P'},
    7:  {2: 'P', 3: 'P', 4: 'P', 5: 'P', 6: 'P', 7: 'P', 8: 'H', 9: 'H', 10: 'H', 11: 'H'},
    6:  {2: 'P', 3: 'P', 4: 'P', 5: 'P', 6: 'P', 7: 'H', 8: 'H', 9: 'H', 10: 'H', 11: 'H'},
    5:  {2: 'Dh', 3: 'Dh', 4: 'Dh', 5: 'Dh', 6: 'Dh', 7: 'Dh', 8: 'Dh', 9: 'Dh', 10: 'H', 11: 'H'},
    4:  {2: 'H', 3: 'H', 4: 'H', 5: 'P', 6: 'P', 7: 'H', 8: 'H', 9: 'H', 10: 'H', 11: 'H'},
    3:  {2: 'P', 3: 'P', 4: 'P', 5: 'P', 6: 'P', 7: 'P', 8: 'H', 9: 'H', 10: 'H', 11: 'H'},
    2:  {2: 'P', 3: 'P', 4: 'P', 5: 'P', 6: 'P', 7: 'P', 8: 'H', 9: 'H', 10: 'H', 11: 'H'},
    10: {2: 'S', 3: 'S', 4: 'S', 5: 'S', 6: 'S', 7: 'S', 8: 'S', 9: 'S', 10: 'S', 11: 'S'},
}


class Shoe:
    def __init__(self, num_decks=None, rng=None):
        self.num_decks = NUM_DECKS if num_decks is None else num_decks
        self.rng = rng or random.Random()
        self.cards = []
        self.reshuffle_count = 0
        self.running_count = 0
        self._new_shoe()

    def _new_shoe(self):
        self.cards = RANK_VALUES * 4 * self.num_decks  # 13 ranks * 4 suits * num_decks
        self.rng.shuffle(self.cards)
        self.reshuffle_count += 1
        self.running_count = 0   # counting always restarts fresh after a shuffle

    def needs_reshuffle(self):
        return len(self.cards) < RESHUFFLE_CUTOFF

    def maybe_reshuffle(self):
        if self.needs_reshuffle():
            self._new_shoe()

    def draw(self, count=True):
        # `count=False` is used for the dealer's face-down hole card, which
        # a real player hasn't seen yet and so can't include in their count.
        # It's added to the running count later via reveal(), once/if it's
        # actually turned face-up.
        if not self.cards:
            self._new_shoe()
        card = self.cards.pop()
        if count and STRATEGY_MODE != 'basic':
            self.running_count += COUNT_SYSTEMS[STRATEGY_MODE][card]
        return card

    def reveal(self, card):
        if STRATEGY_MODE != 'basic':
            self.running_count += COUNT_SYSTEMS[STRATEGY_MODE][card]

    def true_count(self):
        # True count = running count / decks remaining. Normalizing by decks
        # remaining is what makes the count comparable across different
        # amounts of shoe depletion.
        if STRATEGY_MODE == 'basic':
            return 0.0
        decks_remaining = max(len(self.cards) / 52, MIN_DECKS_REMAINING)
        return self.running_count / decks_remaining


def bet_size_for_true_count(tc):
    # Flat 1 unit with no count information. With a count, ramp the bet up
    # by 1 unit per true-count point above 1, capped at BET_SPREAD_MAX -- a
    # simple, commonly-taught linear bet spread.
    if STRATEGY_MODE == 'basic':
        return 1.0
    return min(max(tc - 1.0, 1.0), BET_SPREAD_MAX)


def hand_value(cards):
    # An Ace counts as 11 until that would bust the hand, at which point
    # it converts to 1 (subtract 10 from the total). is_soft is True as
    # long as at least one Ace is still being counted as 11.
    total = sum(cards)
    aces = cards.count(11)
    while total > 21 and aces:
        total -= 10
        aces -= 1
    is_soft = aces > 0
    return total, is_soft


def is_blackjack(cards):
    return len(cards) == 2 and sum(cards) == 21


def pair_action(card_value, dealer_up, true_count=0.0):
    if STRATEGY_MODE != 'basic':
        dev = INDEX_PLAYS.get(('pair', card_value, dealer_up))
        if dev and true_count >= dev[0]:
            return dev[1]
    return PAIR_TOTALS[card_value][dealer_up]


def get_action(total, is_soft, dealer_up, true_count=0.0):
    if not is_soft and STRATEGY_MODE != 'basic':
        dev = INDEX_PLAYS.get(('hard', total, dealer_up))
        if dev and true_count >= dev[0]:
            return dev[1]

    if is_soft:
        if total >= 20:
            return 'S'
        if total in SOFT_TOTALS:
            return SOFT_TOTALS[total][dealer_up]
        return 'H'
    else:
        if total >= 17:
            return 'S'
        if total in HARD_TOTALS:
            return HARD_TOTALS[total][dealer_up]
        return 'H'


def play_hand(hand, dealer_up, shoe, pending, completed, hand_count):
    cards = hand['cards']
    tc = shoe.true_count()

    if len(cards) == 2 and cards[0] == cards[1] and hand_count[0] < MAX_SPLIT_HANDS:
        if pair_action(cards[0], dealer_up, tc) == 'P':
            hand_count[0] += 1
            is_aces = (cards[0] == 11)
            for _ in range(2):
                pending.append({
                    'cards': [cards[0], shoe.draw()],
                    'bet': hand['bet'],
                    'from_aces': is_aces,
                    'can_surrender': False,
                })
            return

    if hand.get('from_aces'):
        completed.append(hand)
        return

    while True:
        total, soft = hand_value(cards)
        if total > 21:
            hand['result'] = 'bust'
            completed.append(hand)
            return
        if total >= 21:
            completed.append(hand)
            return

        tc = shoe.true_count()   # recompute: more cards may have been drawn since the last check
        first_decision = (len(cards) == 2 and not hand.get('has_hit', False))
        can_double = first_decision
        can_surrender = first_decision and hand.get('can_surrender', True)

        action = get_action(total, soft, dealer_up, tc)

        if action == 'Rh':
            if can_surrender:
                hand['result'] = 'surrender'
                completed.append(hand)
                return
            action = 'H'

        if action in ('Dh', 'Ds'):
            if can_double:
                hand['bet'] *= 2
                cards.append(shoe.draw())
                hand['doubled'] = True
                total, _ = hand_value(cards)
                if total > 21:
                    hand['result'] = 'bust'
                completed.append(hand)
                return
            action = 'H' if action == 'Dh' else 'S'

        if action == 'S':
            completed.append(hand)
            return

        cards.append(shoe.draw())
        hand['has_hit'] = True


def play_all_hands(initial_cards, dealer_up, shoe, initial_bet=1.0):
    pending = deque([{'cards': initial_cards, 'bet': initial_bet, 'from_aces': False, 'can_surrender': True}])
    completed = []
    hand_count = [1]
    while pending:
        play_hand(pending.popleft(), dealer_up, shoe, pending, completed, hand_count)
    return completed, hand_count[0]


def play_dealer(cards, shoe):
    while True:
        total, soft = hand_value(cards)
        if total > 21:
            return cards
        if total >= 18 or (total == 17 and not (soft and DEALER_HITS_SOFT_17)):
            return cards
        cards.append(shoe.draw())


def simulate_round(shoe, stats):
    shoe.maybe_reshuffle()

    # The bet is placed using the count as of the END of the previous round
    # (before any of this round's cards are seen) -- that's the information
    # a real player actually has when they put chips out.
    bet = bet_size_for_true_count(shoe.true_count())

    player_cards = [shoe.draw(), shoe.draw()]
    dealer_up = shoe.draw()
    dealer_hole = shoe.draw(count=False)   # face-down: not counted until revealed
    dealer_cards = [dealer_up, dealer_hole]

    stats['rounds'] += 1

    # Count as of everything visible so far (own 2 cards + dealer's up-card).
    decision_tc = shoe.true_count()

    player_bj = is_blackjack(player_cards)

    insurance_net = 0.0
    if STRATEGY_MODE != 'basic' and dealer_up == 11:
        if decision_tc >= INSURANCE_INDEX.get(STRATEGY_MODE, 3):
            stats['insurance_taken'] += 1
            # Insurance is a side bet of half the main bet, paid 2:1.
            if is_blackjack(dealer_cards):
                insurance_net += 1.0 * bet
            else:
                insurance_net -= 0.5 * bet

    # A natural blackjack is only possible if the dealer's up-card is a
    # ten-value card or an Ace, so the hole card only needs checking then.
    if dealer_up in (10, 11):
        dealer_bj = is_blackjack(dealer_cards)
        if dealer_bj:
            shoe.reveal(dealer_hole)
            stats['dealer_blackjacks'] += 1
            stats['total_hands'] += 1
            if player_bj:
                stats['player_blackjacks'] += 1
                stats['pushes'] += 1
                return insurance_net, bet
            stats['losses'] += 1
            return insurance_net - bet, bet
        if player_bj:
            stats['player_blackjacks'] += 1
            stats['wins'] += 1
            stats['total_hands'] += 1
            return insurance_net + 1.5 * bet, bet
    else:
        if player_bj:
            stats['player_blackjacks'] += 1
            stats['wins'] += 1
            stats['total_hands'] += 1
            return insurance_net + 1.5 * bet, bet

    completed_hands, n_hands = play_all_hands(player_cards, dealer_up, shoe, bet)
    if n_hands > 1:
        stats['split_rounds'] += 1

    active_hands = [h for h in completed_hands if h.get('result') not in ('bust', 'surrender')]
    dealer_total = None
    if active_hands:
        # The dealer only draws further cards if at least one player hand
        # is still live. The hole card is only turned face-up (and only
        # then added to the count) at this point.
        shoe.reveal(dealer_hole)
        stats['dealer_hands_played'] += 1
        play_dealer(dealer_cards, shoe)
        dealer_total, _ = hand_value(dealer_cards)
        if dealer_total > 21:
            stats['dealer_busts'] += 1

    total_net = insurance_net
    total_wagered = 0.0
    for h in completed_hands:
        stats['total_hands'] += 1
        total_wagered += h['bet']
        if h.get('doubled'):
            stats['doubles'] += 1
        result = h.get('result')
        if result == 'surrender':
            stats['surrenders'] += 1
            total_net -= 0.5 * h['bet']
        elif result == 'bust':
            stats['player_busts'] += 1
            stats['losses'] += 1
            total_net -= h['bet']
        else:
            p_total, _ = hand_value(h['cards'])
            if dealer_total is None or dealer_total > 21 or p_total > dealer_total:
                stats['wins'] += 1
                total_net += h['bet']
            elif p_total < dealer_total:
                stats['losses'] += 1
                total_net -= h['bet']
            else:
                stats['pushes'] += 1

    return total_net, total_wagered


def new_stats():
    return {
        'rounds': 0, 'wins': 0, 'losses': 0, 'pushes': 0,
        'player_blackjacks': 0, 'dealer_blackjacks': 0,
        'player_busts': 0, 'dealer_busts': 0, 'dealer_hands_played': 0,
        'split_rounds': 0, 'doubles': 0, 'surrenders': 0, 'total_hands': 0,
        'insurance_taken': 0,
    }


def make_batches():
    unit_word = 'flat bet' if STRATEGY_MODE == 'basic' else 'base (min) bet'
    return [
        {'label': f'Batch 1: 0.5% {unit_word}', 'bet_pct': 0.005, 'color': '#2a6f97'},
        {'label': f'Batch 2: 1% {unit_word}',   'bet_pct': 0.01,  'color': '#e08e1f'},
        {'label': f'Batch 3: 2% {unit_word}',   'bet_pct': 0.02,  'color': '#b3282d'},
    ]


def simulate_bot(bet_size, num_hands, seed):
    """
    simulate_round() returns a result in UNITS of a single base bet (e.g.
    +1.5 for blackjack, -1.0 for a loss, -2.0 for a lost double), with
    splits/doubles already folded in. Dollars = net_units * bet_size.

    In basic-strategy mode a unit is always exactly 1, so bet_size is a
    flat dollar bet. In a counting mode, bet_size_for_true_count() already
    varies the units per hand (1 up to BET_SPREAD_MAX) based on the true
    count, so bet_size here becomes "dollars per unit" and the actual
    dollar bet spreads up automatically as the count rises.

    path[0] is the starting bankroll; path[h] is the bankroll after hand h.
    Once bankroll hits $0 it's floored there and stays flat (no more bets).
    """
    rng = random.Random(seed)
    shoe = Shoe(rng=rng)
    stats = new_stats()

    path = np.empty(num_hands + 1)
    path[0] = STARTING_BANKROLL
    bankroll = STARTING_BANKROLL

    for h in range(num_hands):
        if bankroll <= 0:
            path[h + 1] = 0.0
            continue
        net_units, _wagered_units = simulate_round(shoe, stats)
        bankroll += net_units * bet_size
        if bankroll < 0:
            bankroll = 0.0
        path[h + 1] = bankroll

    return path


def run_all(num_bots, num_hands, base_seed):
    """
    The same per-bot seeds are reused across all three batches, so bot i
    sees identical cards and makes identical decisions in every batch --
    only the bet size differs. Without the $0 bust floor this would make
    (bankroll_batch2[i,h] - 10000) exactly 2x (bankroll_batch1[i,h] - 10000),
    and batch 3's exactly 4x, hand for hand. With the floor on, that only
    holds until the first bust; print_summary() reports how far the actual
    ratio falls short of that as a direct measure of the cost of ruin.
    """
    seed_rng = random.Random(base_seed)
    seeds = [seed_rng.randrange(2**32) for _ in range(num_bots)]

    results = []
    for batch in make_batches():
        bet_size = batch['bet_pct'] * STARTING_BANKROLL
        print(f"Simulating {batch['label']} (${bet_size:.0f}/hand, "
              f"{num_bots} bots x {num_hands:,} hands)...")
        paths = np.empty((num_bots, num_hands + 1))
        for i, seed in enumerate(seeds):
            paths[i] = simulate_bot(bet_size, num_hands, seed)
        results.append({**batch, 'bet_size': bet_size, 'paths': paths})
    return results


def compute_stats(paths):
    """Per-hand mean (the trend) and 10th/90th percentile (the spread most
    bots actually land in, wider than the mean alone suggests) across all
    bots in a batch."""
    return {
        'mean': paths.mean(axis=0),
        'p10': np.percentile(paths, 10, axis=0),
        'p50': np.percentile(paths, 50, axis=0),
        'p90': np.percentile(paths, 90, axis=0),
    }


def print_summary(results):
    print("\n" + "=" * 72)
    print("BANKROLL SIMULATION SUMMARY  (starting bankroll = "
          f"${STARTING_BANKROLL:,.0f}, strategy = {STRATEGY_MODE})")
    print("=" * 72)
    for r in results:
        final = r['paths'][:, -1]
        busted = final == 0.0
        n = len(final)
        print(f"\n{r['label']}  (${r['bet_size']:.0f}/hand, {n} bots, "
              f"{r['paths'].shape[1] - 1:,} hands each)")
        print(f"  Mean final bankroll:         ${final.mean():>10,.2f}")
        print(f"  Median final bankroll:       ${np.median(final):>10,.2f}")
        print(f"  Std dev of final bankroll:   ${final.std(ddof=1):>10,.2f}")
        print(f"  Best / worst bot:            ${final.max():>10,.2f} / "
              f"${final.min():>10,.2f}")
        print(f"  % of bots still ahead:       {(final > STARTING_BANKROLL).mean():>9.1%}")
        print(f"  % of bots that went bust ($0): {busted.mean():>7.1%}", end="")
        if busted.any():
            bust_hands = [bust_hand(p) for p in r['paths'][busted]]
            print(f"   (avg. hand of ruin: {np.mean(bust_hands):,.0f})")
        else:
            print()

    # With no bust floor, batch 2's mean move would be exactly 2x batch 1's
    # and batch 3's exactly 4x (see run_all()); the gap from that ratio
    # below is the cost of ruin.
    base_move = results[0]['paths'][:, -1].mean() - STARTING_BANKROLL
    print("\nScaling check (would be exactly x2 / x4 with no bust floor):")
    for r in results:
        move = r['paths'][:, -1].mean() - STARTING_BANKROLL
        ratio = move / base_move if base_move else float('nan')
        print(f"  {r['label']:<28} mean move = {move:+10,.2f}   "
              f"(x{ratio:.2f} vs. Batch 1)")
    print("=" * 72)


def bust_hand(path):
    """First hand index where bankroll hit exactly $0, or None if never."""
    zero_hands = np.flatnonzero(path == 0.0)
    return int(zero_hands[0]) if zero_hands.size else None


def _dollar_axis(ax):
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"${v:,.0f}"))


def plot_time_series(results, num_hands, zoom_hands, outdir):
    """Top row: full run. Bottom row: first zoom_hands hands. Same y-axis
    scale within each row so the growing spread over time is visible."""
    n_sample_paths = 30

    fig, axes = plt.subplots(2, 3, figsize=(16, 9), sharey='row')
    fig.suptitle("Basic-strategy bankroll over time: same edge, three bet sizes",
                  fontsize=14, fontweight='bold')

    for col, r in enumerate(results):
        stats = compute_stats(r['paths'])
        color = r['color']

        for row, hands_to_show in enumerate((num_hands, zoom_hands)):
            ax = axes[row, col]
            x = np.arange(hands_to_show + 1)

            for i in range(min(n_sample_paths, r['paths'].shape[0])):
                ax.plot(x, r['paths'][i, :hands_to_show + 1],
                        color=color, alpha=0.12, linewidth=0.7)

            ax.fill_between(x, stats['p10'][:hands_to_show + 1],
                             stats['p90'][:hands_to_show + 1],
                             color=color, alpha=0.22,
                             label='10th-90th percentile (noise)')
            ax.plot(x, stats['mean'][:hands_to_show + 1],
                     color=color, linewidth=2.2, label='Mean (trend)')
            ax.axhline(STARTING_BANKROLL, color='gray', linestyle='--',
                       linewidth=1, alpha=0.7)

            if row == 0:
                ax.set_title(f"{r['label']}  (${r['bet_size']:.0f}/hand)")
            if col == 0:
                ax.set_ylabel("Full run" if row == 0 else f"First {zoom_hands} hands")
            if row == 1:
                ax.set_xlabel("Hand number")
            _dollar_axis(ax)
            if row == 0 and col == 2:
                ax.legend(loc='upper right', fontsize=8)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    path = os.path.join(outdir, 'bankroll_time_series.png')
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"Saved {path}")


def plot_final_distribution(results, outdir):
    """Same mean relative edge across batches by construction, but the
    variance of outcomes grows with bet size -- the spread here is that
    variance, visualized directly."""
    fig, ax = plt.subplots(figsize=(9, 6))
    for r in results:
        final = r['paths'][:, -1]
        ax.hist(final, bins=40, alpha=0.55, color=r['color'],
                label=f"{r['label']} (mean ${final.mean():,.0f})")
    ax.axvline(STARTING_BANKROLL, color='black', linestyle='--', linewidth=1,
               label='Starting bankroll')
    ax.set_xlabel("Final bankroll ($)")
    ax.set_ylabel("Number of bots")
    ax.set_title("Distribution of final bankrolls: bigger bets = wider spread")
    _dollar_axis(ax)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"${v:,.0f}"))
    ax.legend(fontsize=9)
    fig.tight_layout()
    path = os.path.join(outdir, 'final_bankroll_distribution.png')
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"Saved {path}")


def plot_single_run_comparison(results, num_hands, outdir):
    """Same per-bot seeds across batches (see run_all()), so bot #0's three
    paths here differ only by the bet-size multiplier applied to identical
    underlying swings."""
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(num_hands + 1)
    for r in results:
        ax.plot(x, r['paths'][0], color=r['color'], linewidth=1.6,
                label=f"{r['label']} (${r['bet_size']:.0f}/hand)")
    ax.axhline(STARTING_BANKROLL, color='gray', linestyle='--', linewidth=1)
    ax.set_xlabel("Hand number")
    ax.set_ylabel("Bankroll ($)")
    ax.set_title("Same exact cards, three bet sizes (bot #0 in every batch)")
    _dollar_axis(ax)
    ax.legend(fontsize=9)
    fig.tight_layout()
    path = os.path.join(outdir, 'single_seed_bet_size_comparison.png')
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"Saved {path}")


def plot_decomposition_grid(results, num_hands, bot_index, outdir):
    """Splits one bot's bankroll path into observed = trend + noise:
    Regular   -- the bot's actual cumulative bankroll.
    Trend     -- straight line at the batch-wide mean per-hand result,
                 i.e. what the edge alone would produce with zero variance.
    Noise     -- the bot's per-hand (non-cumulative) dollar result; roughly
                 independent hand to hand, so it looks like noise around a
                 small mean (the edge) that's invisible at this scale.
    Cumulatively summing Noise (plus its mean) reproduces Regular.
    """
    x = np.arange(num_hands + 1)
    x_hands = np.arange(1, num_hands + 1)  # per-hand (not cumulative) axis

    fig, axes = plt.subplots(3, 3, figsize=(16, 11), sharey='row')
    fig.suptitle("Time-series decomposition per batch: observed = trend + noise",
                  fontsize=14, fontweight='bold')

    for col, r in enumerate(results):
        color = r['color']
        bot_path = r['paths'][bot_index]                 # cumulative bankroll
        per_hand = np.diff(bot_path)                       # dollar result of each hand
        mean_increment = np.diff(r['paths'], axis=1).mean()  # batch-wide edge estimate
        trend_line = STARTING_BANKROLL + mean_increment * x

        ax = axes[0, col]
        ax.plot(x, bot_path, color=color, linewidth=1.2)
        ax.axhline(STARTING_BANKROLL, color='gray', linestyle='--', linewidth=1)
        ax.set_title(f"{r['label']}  (${r['bet_size']:.0f}/hand)")
        if col == 0:
            ax.set_ylabel("Regular\n(observed bankroll)")
        _dollar_axis(ax)

        ax = axes[1, col]
        ax.plot(x, trend_line, color=color, linewidth=2.2)
        ax.axhline(STARTING_BANKROLL, color='gray', linestyle='--', linewidth=1)
        ax.text(0.03, 0.05, f"slope = ${mean_increment:+.3f}/hand",
                transform=ax.transAxes, fontsize=8, color='dimgray')
        if col == 0:
            ax.set_ylabel("Trend\n(avg. edge, no variance)")
        _dollar_axis(ax)

        ax = axes[2, col]
        ax.plot(x_hands, per_hand, color=color, linewidth=0.4, alpha=0.7)
        ax.axhline(mean_increment, color='black', linestyle='--', linewidth=1,
                   label=f"mean = ${mean_increment:+.3f}")
        ax.set_xlabel("Hand number")
        if col == 0:
            ax.set_ylabel("White noise\n(per-hand $ result)")
        ax.legend(loc='upper right', fontsize=7)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    path = os.path.join(outdir, 'decomposition_regular_trend_noise.png')
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"Saved {path}")


def export_decomposition_csv(results, num_hands, bot_index, outdir):
    """Per-hand Regular / Trend / White-noise values for the same bot used
    in plot_decomposition_grid(), one column-group per batch."""
    path = os.path.join(outdir, 'decomposition_data.csv')
    x = np.arange(num_hands + 1)

    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        header = ['hand']
        for r in results:
            tag = r['label'].split(':')[0].replace(' ', '_').lower()
            header += [f'{tag}_observed', f'{tag}_trend', f'{tag}_noise_per_hand']
        writer.writerow(header)

        per_batch = []
        for r in results:
            bot_path = r['paths'][bot_index]
            mean_increment = np.diff(r['paths'], axis=1).mean()
            trend_line = STARTING_BANKROLL + mean_increment * x
            per_hand = np.concatenate(([np.nan], np.diff(bot_path)))
            per_batch.append((bot_path, trend_line, per_hand))

        for h in range(num_hands + 1):
            row = [h]
            for bot_path, trend_line, per_hand in per_batch:
                row += [bot_path[h], trend_line[h], per_hand[h]]
            writer.writerow(row)
    print(f"Saved {path}")


def export_csv(results, num_hands, outdir):
    series_path = os.path.join(outdir, 'time_series_summary.csv')
    with open(series_path, 'w', newline='') as f:
        writer = csv.writer(f)
        header = ['hand']
        for r in results:
            tag = r['label'].split(':')[0].replace(' ', '_').lower()
            header += [f'{tag}_mean', f'{tag}_p10', f'{tag}_p50', f'{tag}_p90']
        writer.writerow(header)

        stats_list = [compute_stats(r['paths']) for r in results]
        for h in range(num_hands + 1):
            row = [h]
            for stats in stats_list:
                row += [stats['mean'][h], stats['p10'][h], stats['p50'][h], stats['p90'][h]]
            writer.writerow(row)
    print(f"Saved {series_path}")

    final_path = os.path.join(outdir, 'final_bankrolls.csv')
    with open(final_path, 'w', newline='') as f:
        writer = csv.writer(f)
        header = ['bot_index'] + [r['label'] for r in results]
        writer.writerow(header)
        num_bots = results[0]['paths'].shape[0]
        for i in range(num_bots):
            writer.writerow([i] + [r['paths'][i, -1] for r in results])
    print(f"Saved {final_path}")


def main():
    global STRATEGY_MODE, NUM_DECKS, DEALER_HITS_SOFT_17, BET_SPREAD_MAX

    parser = argparse.ArgumentParser(
        description="Flat-bet bankroll simulator using the basic-strategy / "
                    "card-counting engine also used in blackjack_sim.py")
    parser.add_argument('--hands', type=int, default=NUM_HANDS,
                         help=f'hands played per bot (default: {NUM_HANDS})')
    parser.add_argument('--bots', type=int, default=NUM_BOTS,
                         help=f'independent bots simulated per batch (default: {NUM_BOTS})')
    parser.add_argument('--seed', type=int, default=BASE_SEED,
                         help=f'base random seed, for reproducible runs (default: {BASE_SEED})')
    parser.add_argument('--strategy', choices=['basic', 'hilo', 'omega2'], default=STRATEGY_MODE,
                         help=f'basic strategy, or Hi-Lo / Omega II count-based play with a '
                              f'bet spread (default: {STRATEGY_MODE})')
    parser.add_argument('--decks', type=int, default=NUM_DECKS,
                         help=f'number of decks in the shoe (default: {NUM_DECKS})')
    parser.add_argument('--dealer-rule', choices=['s17', 'h17'], default='s17',
                         help="dealer stands (s17) or hits (h17) on soft 17 (default: s17)")
    parser.add_argument('--bet-spread-max', type=int, default=BET_SPREAD_MAX,
                         help=f'max bet size in units when a counting strategy is active '
                              f'(default: {BET_SPREAD_MAX})')
    parser.add_argument('--zoom-hands', type=int, default=ZOOM_HANDS,
                         help=f'hands shown in the "short-term" zoomed panel (default: {ZOOM_HANDS})')
    parser.add_argument('--decomposition-bot', type=int, default=DECOMPOSITION_BOT,
                         help='bot index used for the regular/trend/noise charts '
                              f'(default: {DECOMPOSITION_BOT})')
    parser.add_argument('--outdir', type=str, default=OUTPUT_DIR,
                         help=f'directory for charts/CSVs (default: {OUTPUT_DIR})')
    parser.add_argument('--no-plot', action='store_true',
                         help='skip chart generation (summary + CSV only)')
    args = parser.parse_args()

    STRATEGY_MODE = args.strategy
    NUM_DECKS = args.decks
    DEALER_HITS_SOFT_17 = (args.dealer_rule == 'h17')
    BET_SPREAD_MAX = args.bet_spread_max

    os.makedirs(args.outdir, exist_ok=True)
    zoom_hands = min(args.zoom_hands, args.hands)
    decomp_bot = min(args.decomposition_bot, args.bots - 1)

    print(f"Simulating {args.bots} bots x {args.hands:,} hands per batch "
          f"({args.bots * args.hands * 3:,} hands total), "
          f"strategy={args.strategy}, {args.decks}-deck {args.dealer_rule}...")
    results = run_all(args.bots, args.hands, args.seed)

    print_summary(results)
    export_csv(results, args.hands, args.outdir)
    export_decomposition_csv(results, args.hands, decomp_bot, args.outdir)

    if not args.no_plot:
        plot_decomposition_grid(results, args.hands, decomp_bot, args.outdir)
        plot_time_series(results, args.hands, zoom_hands, args.outdir)
        plot_final_distribution(results, args.outdir)
        plot_single_run_comparison(results, args.hands, args.outdir)

    print(f"\nAll output written to ./{args.outdir}/")


if __name__ == '__main__':
    main()
