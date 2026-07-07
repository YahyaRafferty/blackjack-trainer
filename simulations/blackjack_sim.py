"""
blackjack_sim.py

Usage
-----
    python blackjack_sim.py                       # basic strategy, default rules
    python blackjack_sim.py --hands 5000000        # simulate 5,000,000 hands
    python blackjack_sim.py --decks 8              # 4, 6, or 8 decks
    python blackjack_sim.py --dealer-rule h17      # dealer hits soft 17 (default: s17, stands)
    python blackjack_sim.py --strategy hilo        # Hi-Lo count-based play + bet spread
    python blackjack_sim.py --strategy omega2      # Omega II count-based play + bet spread
    python blackjack_sim.py --debug                # run the bias/validation checks

Inputs
------
    --dealer-rule   s17 or h17               (default: s17)
    --strategy      basic, hilo, or omega2   (default: basic)
    --decks         any positive integer     (default: 6)
    --hands         any positive integer     (default: 25,000,000)
"""

import argparse
import math
import random
import time
from collections import deque

DEFAULT_NUM_HANDS = 25_000_000
DEFAULT_DEBUG_HANDS = 2_000_000
DEFAULT_SEED = None

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
        # amounts of shoe depletion (a running count of +6 means very
        # different things with 5 decks left vs. with 1 deck left).
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
    # This is what a player actually knows when deciding on insurance and
    # on their first playing decision.
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
        # is still live; a hand that's already bust or surrendered has
        # nothing left to compare against. The hole card is only turned
        # face-up (and only then added to the count) at this point.
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


def run_simulation(num_hands, seed=None, progress_every=None):
    rng = random.Random(seed)
    shoe = Shoe(rng=rng)
    stats = new_stats()

    results = [0.0] * num_hands
    wagered = [0.0] * num_hands

    start = time.time()
    for i in range(num_hands):
        net, total_wagered = simulate_round(shoe, stats)
        results[i] = net
        wagered[i] = total_wagered
        if progress_every and (i + 1) % progress_every == 0:
            elapsed = time.time() - start
            print(f"  ... {i + 1:,} / {num_hands:,} hands "
                  f"({elapsed:,.1f}s elapsed)")

    return results, wagered, stats, shoe


def summarize(results, wagered, stats, shoe, num_hands):
    n = num_hands
    mean_net = sum(results) / n
    # Sample variance (n-1 denominator, Bessel's correction) of the
    # per-hand net result, used to get the standard error of the mean.
    variance = sum((r - mean_net) ** 2 for r in results) / (n - 1)
    std_dev = math.sqrt(variance)
    std_error = std_dev / math.sqrt(n)
    ci95 = 1.96 * std_error   # 95% confidence interval, normal approximation

    mean_wagered = sum(wagered) / n
    house_edge = -mean_net                       # per unit of original bet
    element_of_risk = -mean_net / mean_wagered    # per unit actually wagered (incl. doubles/splits)

    print("\n" + "=" * 60)
    print("SIMULATION RESULTS")
    print("=" * 60)
    print(f"Strategy:                  {STRATEGY_MODE}")
    print(f"Hands played:              {n:,}")
    print(f"Shoe reshuffles:           {shoe.reshuffle_count:,}")
    print(f"Mean net result / hand:    {mean_net:+.5f} units")
    print(f"Std dev / hand:            {std_dev:.5f}")
    print(f"Std error of the mean:     {std_error:.6f}")
    print(f"95% CI on mean net:        [{mean_net - ci95:+.5f}, {mean_net + ci95:+.5f}]")
    print(f"House edge (per orig bet): {house_edge * 100:+.4f}%")
    print(f"Element of risk:           {element_of_risk * 100:+.4f}%  "
          f"(avg wagered/hand = {mean_wagered:.4f})")
    print("-" * 60)
    print(f"Win rate:                  {stats['wins'] / n:.4%}")
    print(f"Loss rate:                 {stats['losses'] / n:.4%}")
    print(f"Push rate:                 {stats['pushes'] / n:.4%}")
    print(f"Player blackjack rate:     {stats['player_blackjacks'] / n:.4%}")
    print(f"Dealer blackjack rate:     {stats['dealer_blackjacks'] / n:.4%}")
    print(f"Player bust rate:          {stats['player_busts'] / n:.4%}")
    print(f"Dealer bust rate (all rounds):        {stats['dealer_busts'] / n:.4%}")
    print(f"Dealer bust rate (rounds dealer played): "
          f"{stats['dealer_busts'] / stats['dealer_hands_played']:.4%}")
    print(f"Rounds with a split:       {stats['split_rounds'] / n:.4%}")
    print(f"Doubles per hand:          {stats['doubles'] / n:.4%}")
    print(f"Surrender rate:            {stats['surrenders'] / n:.4%}")
    print(f"Insurance taken rate:      {stats['insurance_taken'] / n:.4%}")
    print("=" * 60)

    return {
        'mean_net': mean_net, 'std_dev': std_dev, 'std_error': std_error,
        'ci95': ci95, 'house_edge': house_edge, 'element_of_risk': element_of_risk,
        'mean_wagered': mean_wagered,
    }


def run_debug_checks(num_hands=DEFAULT_DEBUG_HANDS, seed=12345):
    print("\n" + "=" * 60)
    print(f"DEBUG / BIAS-CHECK RUN  ({num_hands:,} hands, seed={seed}, strategy={STRATEGY_MODE})")
    print("=" * 60)

    rng = random.Random(seed)
    shoe = Shoe(rng=rng)
    stats = new_stats()
    results, wagered = [], []

    card_counts = {v: 0 for v in set(RANK_VALUES)}
    original_draw = shoe.draw

    def counting_draw(count=True):
        c = original_draw(count=count)
        card_counts[c] += 1
        return c
    shoe.draw = counting_draw

    for _ in range(num_hands):
        net, tw = simulate_round(shoe, stats)
        results.append(net)
        wagered.append(tw)

    n = num_hands
    total_cards = sum(card_counts.values())

    print("\n[1] Card distribution check (raw shoe should be unbiased):")
    # 13 ranks per suit: ranks 2-9 and Ace each have probability 1/13,
    # the ten-value group (10/J/Q/K) has probability 4/13.
    expected_share = {2: 1/13, 3: 1/13, 4: 1/13, 5: 1/13, 6: 1/13, 7: 1/13,
                       8: 1/13, 9: 1/13, 10: 4/13, 11: 1/13}
    for v in sorted(card_counts):
        observed_share = card_counts[v] / total_cards
        print(f"    value {v:>2}: observed {observed_share:6.4f}  "
              f"expected {expected_share[v]:6.4f}  "
              f"(diff {observed_share - expected_share[v]:+.4f})")

    rules_desc = (
        f"{NUM_DECKS}D "
        f"{'H17' if DEALER_HITS_SOFT_17 else 'S17'} DAS LS "
        f"{'no-resplit' if MAX_SPLIT_HANDS <= 2 else f'resplit to {MAX_SPLIT_HANDS} hands'}, "
        f"{STRATEGY_MODE} strategy"
    )
    print(f"\n[2] Known-benchmark comparisons (approximate published values "
          f"for {rules_desc}):")
    bj_rate = stats['player_blackjacks'] / n
    dealer_bust = stats['dealer_busts'] / stats['dealer_hands_played']
    print(f"    Player blackjack rate:  {bj_rate:.4%}   (theory ~4.75-4.83%)")
    print(f"    Dealer bust rate:       {dealer_bust:.4%}   (published ~28-29% figures "
          f"assume the dealer\n"
          f"                            always draws out their hand; this dealer bust "
          f"rate is\n"
          f"                            conditioned on rounds where the dealer actually "
          f"played, i.e.\n"
          f"                            at least one player hand was still live, which "
          f"shifts it upward)")

    mean_net = sum(results) / n
    std_dev = math.sqrt(sum((r - mean_net) ** 2 for r in results) / (n - 1))
    std_error = std_dev / math.sqrt(n)
    house_edge = -mean_net
    print(f"    House edge:             {house_edge * 100:+.4f}%   "
          f"(published basic-strategy house edge for these rules is "
          f"roughly -0.35% to -0.55%;\n"
          f"                            counting strategies should show a smaller "
          f"or reversed edge here,\n"
          f"                            driven mostly by the bet spread rather than "
          f"the play deviations)")
    print(f"    95% CI on house edge:   "
          f"[{(house_edge - 1.96*std_error) * 100:+.4f}%, "
          f"{(house_edge + 1.96*std_error) * 100:+.4f}%]")

    print("\n[3] Sanity checks that should always hold exactly:")
    accounted = stats['wins'] + stats['losses'] + stats['pushes'] + stats['surrenders']
    if accounted == stats['total_hands']:
        print(f"    wins+losses+pushes+surrenders == total hands resolved: "
              f"{accounted:,} == {stats['total_hands']:,}  OK")
    else:
        print(f"    MISMATCH: {accounted:,} vs {stats['total_hands']:,} -- investigate!")
    if stats['total_hands'] >= n:
        print(f"    total hands ({stats['total_hands']:,}) >= rounds played "
              f"({n:,}), as expected once splits are included: OK")
    else:
        print(f"    MISMATCH: total hands {stats['total_hands']:,} < rounds "
              f"played {n:,} -- investigate!")
    print(f"    Reshuffle count:        {shoe.reshuffle_count}")

    print("=" * 60)


def main():
    global NUM_DECKS, DEALER_HITS_SOFT_17, MAX_SPLIT_HANDS, RESHUFFLE_CUTOFF
    global STRATEGY_MODE, BET_SPREAD_MAX

    parser = argparse.ArgumentParser(description="Blackjack basic-strategy / card-counting EV simulator")
    parser.add_argument('--hands', type=int, default=DEFAULT_NUM_HANDS,
                         help=f'number of hands to simulate (default: {DEFAULT_NUM_HANDS:,})')
    parser.add_argument('--decks', type=int, default=NUM_DECKS,
                         help=f'number of decks in the shoe (default: {NUM_DECKS})')
    parser.add_argument('--dealer-rule', choices=['s17', 'h17'], default='s17',
                         help="dealer stands (s17) or hits (h17) on soft 17 (default: s17)")
    parser.add_argument('--strategy', choices=['basic', 'hilo', 'omega2'], default='basic',
                         help="basic strategy (no counting), or Hi-Lo / Omega II "
                              "count-based play with a bet spread (default: basic)")
    parser.add_argument('--bet-spread-max', type=int, default=BET_SPREAD_MAX,
                         help=f'max bet size in units when a counting strategy is active '
                              f'(default: {BET_SPREAD_MAX})')
    parser.add_argument('--max-split-hands', type=int, default=MAX_SPLIT_HANDS,
                         help=f'max hands a round can produce via splitting (default: {MAX_SPLIT_HANDS})')
    parser.add_argument('--reshuffle-cutoff', type=int, default=RESHUFFLE_CUTOFF,
                         help=f'reshuffle once fewer than this many cards remain (default: {RESHUFFLE_CUTOFF})')
    parser.add_argument('--seed', type=int, default=DEFAULT_SEED,
                         help='random seed for reproducibility (default: random)')
    parser.add_argument('--debug', action='store_true',
                         help='run the bias/validation check suite instead of the main sim')
    parser.add_argument('--debug-hands', type=int, default=DEFAULT_DEBUG_HANDS,
                         help=f'number of hands used by --debug (default: {DEFAULT_DEBUG_HANDS:,})')
    args = parser.parse_args()

    NUM_DECKS = args.decks
    DEALER_HITS_SOFT_17 = (args.dealer_rule == 'h17')
    MAX_SPLIT_HANDS = args.max_split_hands
    RESHUFFLE_CUTOFF = args.reshuffle_cutoff
    STRATEGY_MODE = args.strategy
    BET_SPREAD_MAX = args.bet_spread_max

    if args.debug:
        run_debug_checks(num_hands=args.debug_hands, seed=args.seed or 12345)
        return

    rules_desc = f"{NUM_DECKS}-deck {args.dealer_rule.upper()} {STRATEGY_MODE} strategy"
    print(f"Running {args.hands:,} hands of {rules_desc} blackjack...")
    results, wagered, stats, shoe = run_simulation(
        args.hands, seed=args.seed, progress_every=max(args.hands // 10, 1)
    )
    summarize(results, wagered, stats, shoe, args.hands)


if __name__ == '__main__':
    main()
