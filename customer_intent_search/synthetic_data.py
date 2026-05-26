"""
Synthetic Customer Support Dataset Generator
=============================================
Generates a realistic customer support intent dataset that mirrors the
structure of the CLINC OOS dataset. Used when HuggingFace Hub is not
accessible; when running with internet access, the real CLINC OOS dataset
is used instead.

The synthetic dataset covers 20 banking / fintech support intents across
5 domains: banking, cards, travel, utility, and small_talk.
Each intent has 150 training and 30 test query variations.

HuggingFace equivalent: https://huggingface.co/datasets/clinc/clinc_oos
"""
import random
from typing import Dict, List, Tuple

SEED = 42

# ── Intent definitions ────────────────────────────────────────────────────────
# Each intent has: template patterns + slot fillers used to generate queries

INTENTS: Dict[str, Dict] = {
    # BANKING domain
    0: {
        "name": "balance",
        "domain": "banking",
        "templates": [
            "What is my {account} balance?",
            "How much {money} do I have in my account?",
            "Can you tell me my current balance?",
            "Check my {account} balance please",
            "What's the balance on my {account}?",
            "I need to know my account balance",
            "Show me how much money I have",
            "What do I have available in my account?",
            "Can I see my {account} balance?",
            "How much is in my {account}?",
        ],
        "slots": {
            "account": ["savings", "checking", "current", "main", ""],
            "money": ["money", "funds", "cash", ""],
        }
    },
    1: {
        "name": "transfer",
        "domain": "banking",
        "templates": [
            "I want to transfer {amount} to {recipient}",
            "How do I send money to {recipient}?",
            "Can I transfer funds between my accounts?",
            "Send {amount} from my account to {recipient}",
            "I need to move money to {recipient}",
            "Transfer {amount} to my {recipient}",
            "How do I wire money to someone?",
            "I'd like to send {amount} to a friend",
            "Can you help me transfer {amount}?",
            "Move {amount} to my {recipient} account",
        ],
        "slots": {
            "amount": ["$100", "$500", "$50", "some money", "funds", "$200"],
            "recipient": ["friend", "family member", "another account", "my savings", "someone"],
        }
    },
    2: {
        "name": "transactions",
        "domain": "banking",
        "templates": [
            "Show me my recent transactions",
            "What are my last {n} transactions?",
            "I want to see my transaction history",
            "Can I view my recent purchases?",
            "What did I spend money on recently?",
            "List my recent transactions",
            "Show me my spending history",
            "What transactions have I made this {period}?",
            "I need to review my recent charges",
            "Can you show me my account activity?",
        ],
        "slots": {
            "n": ["5", "10", "20", "few"],
            "period": ["week", "month", "year"],
        }
    },
    3: {
        "name": "freeze_account",
        "domain": "banking",
        "templates": [
            "I need to freeze my account",
            "Please lock my {card} immediately",
            "Block my account right now",
            "How do I freeze my debit card?",
            "I want to temporarily lock my account",
            "Freeze my {card} please",
            "Can you block my account for security?",
            "I need to lock my account ASAP",
            "How do I put a hold on my account?",
            "Disable my account temporarily",
        ],
        "slots": {
            "card": ["card", "debit card", "credit card", "account"],
        }
    },
    4: {
        "name": "pay_bill",
        "domain": "banking",
        "templates": [
            "I want to pay my {bill} bill",
            "How do I pay a {bill}?",
            "Can I pay my {bill} through the app?",
            "Help me pay my {bill} bill",
            "I need to make a bill payment",
            "Pay my {bill} from my account",
            "How do I set up automatic bill payment?",
            "I'd like to pay my {bill} online",
            "Process my {bill} payment please",
            "Can you pay my {bill} for me?",
        ],
        "slots": {
            "bill": ["electricity", "water", "phone", "internet", "credit card", "utility", "rent"],
        }
    },

    # CARDS domain
    5: {
        "name": "report_lost_card",
        "domain": "cards",
        "templates": [
            "I lost my {card}",
            "My {card} was stolen",
            "I can't find my {card} anywhere",
            "Someone stole my {card}",
            "I think I lost my {card}",
            "My {card} is missing",
            "I need to report a lost {card}",
            "I've misplaced my {card}",
            "Please report my {card} as lost",
            "How do I report a stolen {card}?",
        ],
        "slots": {
            "card": ["debit card", "credit card", "card", "bank card"],
        }
    },
    6: {
        "name": "card_declined",
        "domain": "cards",
        "templates": [
            "My card was declined",
            "Why was my {card} declined?",
            "My payment was rejected",
            "My {card} isn't working",
            "The transaction was declined",
            "I got a card declined error",
            "My {card} was refused at checkout",
            "Why is my card not being accepted?",
            "My card keeps getting declined",
            "Why did my payment fail?",
        ],
        "slots": {
            "card": ["debit card", "credit card", "card", "visa"],
        }
    },
    7: {
        "name": "credit_score",
        "domain": "cards",
        "templates": [
            "What is my credit score?",
            "How can I improve my credit rating?",
            "Check my credit score please",
            "How do I build up my credit?",
            "What factors affect my credit score?",
            "I want to see my credit report",
            "Can you show me my credit rating?",
            "How do I increase my credit score?",
            "Why is my credit score low?",
            "I need help with my credit score",
        ],
        "slots": {}
    },
    8: {
        "name": "credit_limit",
        "domain": "cards",
        "templates": [
            "Can I increase my credit limit?",
            "What is my credit limit?",
            "How do I get a higher credit limit?",
            "I'd like to request a credit limit increase",
            "What's the maximum credit limit I can get?",
            "How can I raise my credit limit?",
            "Check my available credit",
            "I need a higher spending limit",
            "Can you increase my credit line?",
            "What's my current credit limit?",
        ],
        "slots": {}
    },
    9: {
        "name": "new_card",
        "domain": "cards",
        "templates": [
            "I want to get a new {card}",
            "How do I apply for a {card}?",
            "I'd like to order a replacement card",
            "Can I get a new debit card?",
            "Apply for a new credit card",
            "I need a new {card} issued",
            "How do I get a replacement {card}?",
            "I want to request a new card",
            "Order me a new {card} please",
            "I need to replace my {card}",
        ],
        "slots": {
            "card": ["debit card", "credit card", "card", "Visa card"],
        }
    },

    # TRAVEL domain
    10: {
        "name": "exchange_rate",
        "domain": "travel",
        "templates": [
            "What is the exchange rate for {currency}?",
            "How much is {currency} to {currency2}?",
            "What's today's {currency} exchange rate?",
            "Can I see the current exchange rates?",
            "Convert {currency} to {currency2} please",
            "What's the rate for converting to {currency}?",
            "How much does {currency} cost today?",
            "I need the forex rate for {currency}",
            "What are your exchange rates?",
            "Give me the {currency} exchange rate",
        ],
        "slots": {
            "currency": ["USD", "EUR", "GBP", "JPY", "CAD", "dollars", "euros", "pounds"],
            "currency2": ["EUR", "USD", "GBP", "local currency"],
        }
    },
    11: {
        "name": "travel_notification",
        "domain": "travel",
        "templates": [
            "I'm traveling to {country} next {period}",
            "Let my bank know I'm going to {country}",
            "I need to set up a travel notification",
            "I'll be in {country} from {date}",
            "Please note my travel to {country}",
            "I'm going abroad to {country}",
            "Set a travel alert for my trip to {country}",
            "Notify my bank about my trip to {country}",
            "I'm traveling internationally to {country}",
            "Update my travel plans to {country}",
        ],
        "slots": {
            "country": ["France", "Japan", "Canada", "Mexico", "Germany", "Italy", "Spain", "Europe"],
            "period": ["week", "month", "weekend"],
            "date": ["next week", "Monday", "the 15th"],
        }
    },
    12: {
        "name": "international_fees",
        "domain": "travel",
        "templates": [
            "Are there fees for using my card abroad?",
            "What are the international transaction fees?",
            "How much do you charge for foreign transactions?",
            "Will I be charged fees for using my card in {country}?",
            "What are the ATM fees internationally?",
            "Do you charge for overseas withdrawals?",
            "What's the foreign transaction fee?",
            "How much does it cost to use my card abroad?",
            "Are there charges for international payments?",
            "Tell me about your international fees",
        ],
        "slots": {
            "country": ["Europe", "Japan", "Canada", "Mexico", "abroad"],
        }
    },
    13: {
        "name": "book_flight",
        "domain": "travel",
        "templates": [
            "Book a flight to {destination}",
            "I want to fly to {destination}",
            "Find me flights to {destination}",
            "Can I book a flight to {destination}?",
            "I need to travel to {destination} by plane",
            "Book the cheapest flight to {destination}",
            "Find flights from here to {destination}",
            "I'd like to book a round trip to {destination}",
            "What flights are available to {destination}?",
            "Help me book a flight to {destination}",
        ],
        "slots": {
            "destination": ["New York", "London", "Paris", "Tokyo", "LA", "Miami", "Chicago"],
        }
    },

    # UTILITY domain
    14: {
        "name": "spending_history",
        "domain": "utility",
        "templates": [
            "Show me my spending this {period}",
            "How much have I spent on {category}?",
            "What are my biggest expenses this {period}?",
            "Give me a spending breakdown",
            "Where am I spending the most money?",
            "Analyze my spending habits",
            "How much did I spend last {period}?",
            "Show me my expenses by category",
            "What's my monthly spending total?",
            "I want to see my spending patterns",
        ],
        "slots": {
            "period": ["month", "week", "year", "quarter"],
            "category": ["food", "entertainment", "shopping", "travel", "dining"],
        }
    },
    15: {
        "name": "direct_deposit",
        "domain": "utility",
        "templates": [
            "How do I set up direct deposit?",
            "I want to get my paycheck deposited directly",
            "Set up direct deposit for my salary",
            "How do I enroll in direct deposit?",
            "I want my paycheck automatically deposited",
            "Can you set up direct deposit for me?",
            "Give me my direct deposit information",
            "What's my routing number for direct deposit?",
            "How does direct deposit work?",
            "I need to add direct deposit to my account",
        ],
        "slots": {}
    },
    16: {
        "name": "interest_rates",
        "domain": "utility",
        "templates": [
            "What's the interest rate on my {account}?",
            "How much interest am I earning?",
            "What are your current savings interest rates?",
            "What's the APY on my {account}?",
            "Tell me about your interest rates",
            "How much interest will I earn on my savings?",
            "What's the rate on my {account}?",
            "I want to know my interest rate",
            "How is interest calculated on my account?",
            "What interest rates do you offer?",
        ],
        "slots": {
            "account": ["savings account", "checking account", "CD", "money market"],
        }
    },

    # SMALL TALK domain
    17: {
        "name": "greeting",
        "domain": "small_talk",
        "templates": [
            "Hello!",
            "Hi there",
            "Good morning",
            "Hey, how are you?",
            "Good afternoon",
            "Good evening",
            "Hi, I need some help",
            "Hello, can you assist me?",
            "Hey there!",
            "Good day",
        ],
        "slots": {}
    },
    18: {
        "name": "thank_you",
        "domain": "small_talk",
        "templates": [
            "Thank you so much!",
            "Thanks for your help",
            "That was very helpful, thank you",
            "Thanks!",
            "Much appreciated",
            "Thank you for the assistance",
            "Great help, thank you",
            "You've been very helpful, thanks",
            "Thanks a lot!",
            "I appreciate your help",
        ],
        "slots": {}
    },
    19: {
        "name": "cancel_reservation",
        "domain": "travel",
        "templates": [
            "I need to cancel my {booking} reservation",
            "Cancel my booking for {destination}",
            "How do I cancel a reservation?",
            "I want to cancel my {booking}",
            "Please cancel my {booking} reservation",
            "Can I get a refund for my cancelled {booking}?",
            "Cancel my trip to {destination}",
            "I need to cancel my upcoming {booking}",
            "How do I get a refund on a cancelled {booking}?",
            "Cancel my hotel reservation in {destination}",
        ],
        "slots": {
            "booking": ["flight", "hotel", "car rental", "travel"],
            "destination": ["Paris", "New York", "London", "Tokyo"],
        }
    },
}


def _fill_template(template: str, slots: dict, rng: random.Random) -> str:
    """Fill a template string with random slot values."""
    result = template
    for slot_name, options in slots.items():
        placeholder = "{" + slot_name + "}"
        if placeholder in result and options:
            result = result.replace(placeholder, rng.choice(options), 1)
    # Clean up any remaining unfilled optional placeholders
    import re
    result = re.sub(r'\{[^}]+\}', '', result).strip()
    return result


def _generate_queries(
    intent_config: dict,
    n_queries: int,
    rng: random.Random,
) -> List[str]:
    """Generate `n_queries` realistic queries for an intent."""
    templates = intent_config["templates"]
    slots = intent_config.get("slots", {})
    queries = set()
    attempts = 0
    max_attempts = n_queries * 20

    # First pass: use each template at least once
    for tmpl in templates:
        if len(queries) >= n_queries:
            break
        q = _fill_template(tmpl, slots, rng)
        queries.add(q)

    # Second pass: generate variations via repeated template filling
    while len(queries) < n_queries and attempts < max_attempts:
        tmpl = rng.choice(templates)
        q = _fill_template(tmpl, slots, rng)

        # Add slight variations: capitalization, punctuation
        variations = [
            q,
            q.lower(),
            q.rstrip("?") + "?",
            "Please " + q[0].lower() + q[1:],
            q + " please",
            "Can you help me? " + q,
            "I was wondering: " + q[0].lower() + q[1:],
        ]
        queries.add(rng.choice(variations))
        attempts += 1

    # Fill remaining spots if needed
    result = list(queries)[:n_queries]
    while len(result) < n_queries:
        tmpl = rng.choice(templates)
        result.append(_fill_template(tmpl, slots, rng) + f" (query {len(result)})")

    return result


def generate_synthetic_dataset(
    n_train: int = 150,
    n_val: int = 30,
    n_test: int = 30,
    seed: int = SEED,
) -> Tuple[Dict[int, List[str]], Dict[int, List[str]], Dict[int, List[str]]]:
    """
    Generate a synthetic customer support dataset mirroring CLINC OOS structure.

    Args:
        n_train: queries per intent for training
        n_val: queries per intent for validation
        n_test: queries per intent for testing
        seed: random seed

    Returns:
        train_data, val_data, test_data — each a dict {intent_id: [texts]}
    """
    rng = random.Random(seed)
    train_data, val_data, test_data = {}, {}, {}

    for intent_id, intent_config in INTENTS.items():
        all_queries = _generate_queries(
            intent_config, n_train + n_val + n_test, rng
        )
        # Ensure no overlap between splits
        train_data[intent_id] = all_queries[:n_train]
        val_data[intent_id] = all_queries[n_train: n_train + n_val]
        test_data[intent_id] = all_queries[n_train + n_val:]

    n_intents = len(INTENTS)
    print(f"  Intents: {n_intents}")
    print(f"  Train queries: {n_intents * n_train:,}  ({n_train}/intent)")
    print(f"  Val queries:   {n_intents * n_val:,}  ({n_val}/intent)")
    print(f"  Test queries:  {n_intents * n_test:,}  ({n_test}/intent)")
    print(f"  Domains: banking, cards, travel, utility, small_talk")

    return train_data, val_data, test_data


def get_intent_name(intent_id: int) -> str:
    """Get human-readable name for an intent ID."""
    return INTENTS.get(intent_id, {}).get("name", f"intent_{intent_id}")


def get_intent_domain(intent_id: int) -> str:
    """Get domain name for an intent ID."""
    return INTENTS.get(intent_id, {}).get("domain", "unknown")


if __name__ == "__main__":
    print("Generating synthetic customer support dataset...")
    train, val, test = generate_synthetic_dataset()

    print("\nSample queries per intent:")
    for intent_id in list(INTENTS.keys())[:5]:
        name = get_intent_name(intent_id)
        domain = get_intent_domain(intent_id)
        print(f"\n  [{domain}] Intent {intent_id}: {name}")
        for q in train[intent_id][:4]:
            print(f"    - {q}")
