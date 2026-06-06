"""
templates.py  -  per-intent phrasings, split into train vs test.
================================================================================
Pure data. Each intent has TWO disjoint lists of templates: `train` phrasings and
`test` phrasings. Keeping them different is leakage defence #1 -- the test set
asks for the same intents using sentence shapes the model never trained on.

A {slot} token is looked up in slots.py; a {a|b|c} token is an inline choice.
"""

# templates per intent: separate phrasings for train vs test
TEMPLATES = {
    "smart_home": {
        "train": [
            "turn {on|off} the {room} {fixture}",
            "{switch on|switch off|turn on|turn off} the {fixture} in the {room}",
            "{fixture} {on|off}",
            "dim the {room} {fixture} to {pct} percent",
            "make the {room} {brighter|darker}",
            "set the {room} {fixture} to {warm white|cool white|daylight}",
            # realistic-phrasing augmentation (casual / indirect state)
            "it's too {dark|bright} in the {room}",
            "the {room} is too {dark|bright}",
            "i need {more|less} light in the {room}",
            "can you {brighten|dim} the {room}",
            "give me some light in the {room}",
        ],
        "test": [
            "can you {turn on|turn off} the {room} {fixture}",
            "{brighten|dim} the {fixture} in the {room}",
            "i want the {room} {fixture} {on|off}",
        ],
    },
    "media": {
        "train": [
            "play {genre}",
            "play {artist}",
            "put on {playlist}",
            "play some {genre} {service_phrase}",
            "{pause|stop|resume} the {music|podcast|song}",
            "skip {to the next track|this song|ahead}",
            "turn the volume {up|down}",
            "play {playlist} {service_phrase}",
            # desire/request phrasings (so "i want to listen to X" generalizes —
            # different surface forms than the held-out test template)
            "i'd like to hear some {genre}",
            "i'm in the mood for some {genre}",
            "can you put on some {genre} {service_phrase}",
            "let's hear some {genre}",
            "throw on {playlist}",
            "queue up {artist} {service_phrase}",
            # realistic-phrasing augmentation (casual)
            "put on some {genre}",
            "play me something {upbeat|chill|relaxing}",
            "turn the music {up|down}",
            "gimme some {genre}",
        ],
        "test": [
            "i want to listen to {genre}",
            "can you play {artist} {service_phrase}",
            "go to the {next|previous} {song|track}",
            "turn it {up|down}",
        ],
    },
    "productivity": {
        "train": [
            "schedule a {meeting|call|appointment} with {contact} {when_phrase}",
            "add {a dentist appointment|a meeting|an event} {when_phrase}",
            "what's on my calendar {today|tomorrow|this week}",
            "am i free {when_phrase}",
            "remind me to {reminders}",
            "move my {time} {meeting|call} to {time}",
            "cancel my {time} {meeting|appointment}",
            "what's my {next|first} {meeting|appointment}",
            # realistic-phrasing augmentation (casual)
            "what do i have {today|tomorrow|this week}",
            "don't let me forget to {reminders}",
            "am i busy {when_phrase}",
            "put a {meeting|call} with {contact} on my calendar {when_phrase}",
            "jot down {reminders}",
        ],
        "test": [
            "set up a {meeting|call} with {contact} {when_phrase}",
            "do i have anything {today|tomorrow}",
            "set a reminder to {reminders}",
            "when is my next {meeting|appointment}",
        ],
    },
    "communication": {
        "train": [
            "call {contact}",
            "give {contact} a call",
            "text {contact} {message_topic}",
            "send {contact} a {text|message}",
            "email {contact} {message_topic}",
            "send {contact} an email",
            "{reply to|forward} {contact}'s email",
            "read me my {latest|last} email",
            "check my {inbox|messages}",
            # realistic-phrasing augmentation (casual)
            "give {contact} a ring",
            "shoot {contact} a {text|message}",
            "let {contact} know i'm running late",
            "drop {contact} a line",
        ],
        "test": [
            "write an email to {contact}",
            "any new {emails|messages} from {contact}",
            "call {contact} for me",
            "send a text to {contact}",
        ],
    },
    "timers_alarms": {
        "train": [
            "set a timer for {duration}",
            "set a timer for {timer_label}",
            "set an alarm for {time}",
            "wake me up at {time}",
            "{cancel|stop|pause} {my timer|the timer|the alarm}",
            "{snooze|turn off|dismiss} the alarm",
            "how {much time is|long is} left on my timer",
            # realistic-phrasing augmentation (casual)
            "alert me in {duration}",
            "buzz me in {duration}",
            "set an alarm so i'm up by {time}",
            "give me {duration} on a timer",
        ],
        "test": [
            "start a timer for {duration}",
            "set the alarm for {time}",
            "wake me at {time}",
            "how long left on my {timer|alarm}",
        ],
    },
    "weather": {
        "train": [
            "what's the weather {today|tomorrow|this weekend|tonight}",
            "is it going to {rain|snow|be sunny|be windy} {today|tomorrow|tonight}",
            "what's the {temperature|forecast} {today|tomorrow|this weekend}",
            "do i need {an umbrella|a jacket|sunglasses} {today|tomorrow}",
            "how {hot|cold|windy} is it {today|right now|going to be}",
            "what's the weather like in {city}",
            # realistic-phrasing augmentation (casual)
            "is it {cold|hot|chilly|warm} out {today|right now}",
            "what's the temp {today|tomorrow}",
            "think it'll {rain|snow} {today|later}",
            "how's the weather looking {today|this week}",
            "do i need {a coat|a jacket|sunglasses} for {today|tomorrow}",
        ],
        "test": [
            "will it {rain|snow} {today|tomorrow|this weekend}",
            "is it {hot|cold|sunny} {outside|right now}",
            "what's the forecast for {city}",
            "should i bring {an umbrella|a coat} {today|tomorrow}",
        ],
    },
    "navigation": {
        "train": [
            "navigate to {place}",
            "directions to {place}",
            "how long {to get to|to} {place}",
            "what's the {fastest|best} route to {place}",
            "is there traffic {to|on the way to} {place}",
            "find {a gas station|parking|a coffee shop} near {me|here}",
            "take me to {place}",
            # realistic-phrasing augmentation (casual)
            "what's the quickest way to {place}",
            "is there traffic heading to {place}",
            "get me to {place}",
            "find {a gas station|parking|a coffee shop} around here",
            "where's the nearest {gas station|atm|pharmacy}",
        ],
        "test": [
            "how do i get to {place}",
            "what's traffic like {to|near} {place}",
            "find the nearest {gas station|pharmacy|atm}",
            "route me to {place}",
        ],
    },
    "answers": {
        "train": [
            "what's the capital of {country}",
            "how do you say {word} in {language}",
            "convert {conversion_phrase}",
            "what's {n} percent of {n2}",
            "who {achievement}",
            "how {tall|deep|far|big} is {landmark}",
            "why is {phenomenon}",
            # realistic-phrasing augmentation (diverse knowledge questions)
            "what is {topic}",
            "explain {topic} to me",
            "tell me about {topic}",
            "what does {acronym} stand for",
        ],
        "test": [
            "tell me the capital of {country}",
            "what year did {event}",
            "what's the {height|size} of {landmark}",
            "how do i say {word} in {language}",
        ],
    },
    "none": {
        "train": [
            "order me {food}",
            "book me a {transport}",
            "call me {an uber|a taxi|a cab}",
            "add {food} to my shopping list",
            "{ugh|meh|hmm} {what a day|whatever|never mind}",
            "{gibberish}",
        ],
        "test": [
            "i need a {transport} to the airport",
            "order {food} for delivery",
            "translate this {document|page} into {language}",
        ],
    },
}
