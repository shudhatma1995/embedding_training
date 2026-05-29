import random
from typing import Dict, List, Tuple

# ── Intent metadata ────────────────────────────────────────────
INTENT_NAMES = {
    # orders
    0:  "track_order",
    1:  "cancel_order",
    2:  "modify_order",
    3:  "order_not_received",
    4:  "wrong_item_received",
    # returns
    5:  "initiate_return",
    6:  "refund_status",
    7:  "exchange_item",
    8:  "damaged_item",
    # delivery
    9:  "delivery_address_change",
    10: "delivery_delay",
    11: "express_shipping",
    # payments
    12: "payment_failed",
    13: "apply_promo_code",
    14: "price_match",
    15: "invoice_request",
    # account
    16: "reset_password",
    17: "loyalty_points",
    18: "gift_card_balance",
    19: "contact_support",
}

INTENT_DOMAINS = {
    0: "orders",   1: "orders",   2: "orders",   3: "orders",   4: "orders",
    5: "returns",  6: "returns",  7: "returns",  8: "returns",
    9: "delivery", 10: "delivery", 11: "delivery",
    12: "payments", 13: "payments", 14: "payments", 15: "payments",
    16: "account",  17: "account",  18: "account",  19: "account",
}

# ── Slots (fill-in values for templates) ──────────────────────
SLOTS = {
    "order_id":   ["#ORD-1234", "#ORD-5678", "#ORD-9012", "order 4521", "order number 7823"],
    "item":       ["shoes", "jacket", "phone case", "laptop bag", "dress", "headphones", "watch", "backpack"],
    "recipient":  ["john", "my sister", "my friend", "sarah", "my mom"],
    "amount":     ["$20", "$50", "$100", "20 dollars", "fifty dollars"],
    "date":       ["yesterday", "last week", "3 days ago", "on monday", "last friday"],
    "address":    ["123 Main St", "my new apartment", "my office address", "456 Oak Ave"],
    "promo_code": ["SAVE10", "SUMMER20", "WELCOME15", "DISCOUNT25"],
    "store":      ["Amazon", "Walmart", "Target", "another store"],
    "reason":     ["it doesn't fit", "I changed my mind", "found a better price", "it looks different from the picture"],
}

# ── Templates per intent ───────────────────────────────────────
TEMPLATES = {
    0: [  # track_order
        "where is my {order_id}",
        "can you track my {order_id}",
        "what is the status of my {order_id}",
        "I want to know where my order is",
        "has my {order_id} shipped yet",
        "when will my {order_id} arrive",
        "I need to track my package",
        "give me an update on my {order_id}",
    ],
    1: [  # cancel_order
        "I want to cancel my {order_id}",
        "please cancel {order_id}",
        "can I cancel my order",
        "I need to cancel my recent order",
        "cancel my {order_id} immediately",
        "how do I cancel {order_id}",
        "I changed my mind cancel {order_id}",
    ],
    2: [  # modify_order
        "I want to change my {order_id}",
        "can I modify my order",
        "I need to update the size on {order_id}",
        "change the color on my {order_id}",
        "I ordered the wrong size can you fix it",
        "update my {order_id} please",
        "I want to add an item to {order_id}",
    ],
    3: [  # order_not_received
        "I never received my {order_id}",
        "my order has not arrived",
        "where is my package it was due {date}",
        "I have not received {order_id} yet",
        "my {order_id} is missing",
        "I am still waiting for my order from {date}",
        "my package never showed up",
    ],
    4: [  # wrong_item_received
        "I received the wrong {item}",
        "you sent me the wrong item",
        "this is not what I ordered",
        "I got a {item} but I ordered something else",
        "wrong item in my package",
        "the {item} I received is not correct",
        "I ordered a {item} but got something different",
    ],
    5: [  # initiate_return
        "I want to return my {item}",
        "how do I return {item}",
        "I need to send back my {item}",
        "can I return {item} because {reason}",
        "I would like to return my order",
        "start a return for {order_id}",
        "I want to send {item} back",
    ],
    6: [  # refund_status
        "where is my refund",
        "when will I get my money back",
        "I returned {order_id} where is the refund",
        "has my refund been processed",
        "I have not received my refund yet",
        "how long does a refund take",
        "check my refund status for {order_id}",
    ],
    7: [  # exchange_item
        "I want to exchange my {item}",
        "can I swap my {item} for a different size",
        "I need to exchange {item} because {reason}",
        "how do I exchange my order",
        "swap my {item} for another one",
        "I want a different size can I exchange",
        "exchange {order_id} for a different color",
    ],
    8: [  # damaged_item
        "my {item} arrived damaged",
        "the {item} I received is broken",
        "I got a damaged {item}",
        "my package arrived and the {item} was broken",
        "the {item} is defective",
        "I received a broken {item}",
        "my {item} was damaged during shipping",
    ],
    9: [  # delivery_address_change
        "I need to change my delivery address",
        "can I update the shipping address for {order_id}",
        "change my address to {address}",
        "wrong address on my order can you fix it",
        "update delivery address to {address}",
        "I moved can you ship to {address}",
        "change shipping address on {order_id}",
    ],
    10: [  # delivery_delay
        "my order is delayed",
        "why is my {order_id} taking so long",
        "my package was supposed to arrive {date}",
        "there is a delay on my {order_id}",
        "my order is late",
        "expected delivery was {date} but still nothing",
        "why has my order not arrived yet",
    ],
    11: [  # express_shipping
        "can I get express shipping",
        "how much is overnight shipping",
        "I need my order delivered tomorrow",
        "upgrade to express shipping for {order_id}",
        "do you offer same day delivery",
        "I need fast shipping for {order_id}",
        "what are the express shipping options",
    ],
    12: [  # payment_failed
        "my payment failed",
        "I cannot complete my purchase",
        "my card was declined",
        "payment not going through",
        "I am having trouble paying",
        "my transaction failed",
        "why was my payment rejected",
    ],
    13: [  # apply_promo_code
        "how do I apply promo code {promo_code}",
        "my coupon {promo_code} is not working",
        "I have a discount code {promo_code}",
        "promo code {promo_code} not working",
        "can I use code {promo_code} on my order",
        "apply discount {promo_code} to my cart",
        "my promotional code is not applying",
    ],
    14: [  # price_match
        "I found this cheaper at {store}",
        "can you match the price from {store}",
        "price match with {store}",
        "the same item is cheaper at {store}",
        "do you offer price matching",
        "I want a price match against {store}",
        "found a lower price can you match it",
    ],
    15: [  # invoice_request
        "I need an invoice for {order_id}",
        "can you send me a receipt",
        "I need a tax invoice for my order",
        "please email me the invoice for {order_id}",
        "I need proof of purchase for {order_id}",
        "send me a receipt for my recent order",
        "where can I download my invoice",
    ],
    16: [  # reset_password
        "I forgot my password",
        "how do I reset my password",
        "I cannot log into my account",
        "reset my account password",
        "I am locked out of my account",
        "send me a password reset link",
        "I need to change my password",
    ],
    17: [  # loyalty_points
        "how many loyalty points do I have",
        "check my rewards points",
        "I want to redeem my points",
        "how do I use my loyalty points",
        "my points are missing",
        "how do I earn more points",
        "what can I do with my loyalty points",
    ],
    18: [  # gift_card_balance
        "what is my gift card balance",
        "check my gift card",
        "how much is left on my gift card",
        "I want to use my gift card",
        "my gift card is not working",
        "check balance on gift card",
        "how do I apply my gift card",
    ],
    19: [  # contact_support
        "I need to speak to someone",
        "connect me to an agent",
        "I want to talk to a human",
        "can I speak to customer service",
        "transfer me to support",
        "I need help from a real person",
        "connect me to customer care",
    ],
}

def _fill_template(template: str) -> str:
    result = template
    for slot, values in SLOTS.items():
        placeholder = "{" + slot + "}"
        if placeholder in result:
            result = result.replace(placeholder, random.choice(values))
    
    # Add variation — sometimes capitalize, sometimes add punctuation
    variations = [
        result,
        result.capitalize(),
        result + "?",
        result.capitalize() + "?",
        result + " please",
        result.capitalize() + " please.",
    ]
    return random.choice(variations)

def generate_synthetic_dataset(
    n_train: int = 150,
    n_val: int = 30,
    n_test: int = 30,
    seed: int = 42,
) -> Tuple[Dict, Dict, Dict]:
    random.seed(seed)

    train_data = {}
    val_data = {}
    test_data = {}

    for intent_id, templates in TEMPLATES.items():
        all_queries = []

        # keep generating until we have enough
        while len(all_queries) < n_train + n_val + n_test:
            for template in templates:
                all_queries.append(_fill_template(template))

        # shuffle and split
        random.shuffle(all_queries)
        train_data[intent_id] = all_queries[:n_train]
        val_data[intent_id]   = all_queries[n_train: n_train + n_val]
        test_data[intent_id]  = all_queries[n_train + n_val: n_train + n_val + n_test]

    print(f"  Generated synthetic e-commerce dataset:")
    print(f"  Intents : {len(TEMPLATES)}")
    print(f"  Train   : {len(TEMPLATES) * n_train:,} queries")
    print(f"  Val     : {len(TEMPLATES) * n_val:,} queries")
    print(f"  Test    : {len(TEMPLATES) * n_test:,} queries")

    return train_data, val_data, test_data


def get_intent_name(intent_id: int) -> str:
    return INTENT_NAMES.get(intent_id, f"unknown_{intent_id}")


def get_intent_domain(intent_id: int) -> str:
    return INTENT_DOMAINS.get(intent_id, "unknown")


def get_all_intent_ids() -> List[int]:
    return list(INTENT_NAMES.keys())


def get_intents_by_domain(domain: str) -> List[int]:
    return [id for id, d in INTENT_DOMAINS.items() if d == domain]


# main
if __name__ == "__main__":
    from synthetic_data import generate_synthetic_dataset, get_intent_name, get_intent_domain
    train, val, test = generate_synthetic_dataset()

    # check sizes
    print(len(train[0]))   # 150
    print(len(val[0]))     # 30
    print(len(test[0]))    # 30

    # look at some samples
    print(get_intent_name(0))          # track_order
    print(get_intent_domain(0))        # orders
    print(train[0][:3])                # 3 sample track_order queries
    print(train[5][:3])                # 3 sample initiate_return queries