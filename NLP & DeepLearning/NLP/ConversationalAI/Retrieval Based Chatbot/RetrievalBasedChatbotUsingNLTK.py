from nltk.chat.util import Chat, reflections

# Automated replies defined using regex
pairs = [
    [r"what is your name ?", ["My name is Botty?",]],
    [r"how are you ?", ["I'm fine\nWhat about you ?",]],
    [r"my name is (.*)", ["Hello %1, How are you today ?",]],
    [r"(.*) sorry ", ["Its alright", "Its OK",]],
    [r"hi|hey|hello", ["Hola", "Hey", "Hi"]],
    [r"(.*) age?", ["I'm a computer program \nTime doesn't affect me!",]],
    [r"(.*) created (.*) ?", ["God created me using Python's NLTK library ",]],
    [r"(.*) (location|city|live|place) ?", ["In your mind",]],
    [
        r"how (.*) health(.*)",
        [
            "I'm a computer program, so I'm always healthy except for the occasional bugs ;)",
        ],
    ],
    [r"(.*) (sports|game) ?", ["I'm a very big fan of football",]],
    [r"any favourite player ?", ["Ronaldo. Just kidding its Messi",]],
    [r"(bye|see ya|goodbye)", ["Talk to you soon :) ", "Finally. Good Bye"]],
]

# Chatbot creation using the above defined pairs(and reflections)
chat = Chat(pairs, reflections)

print(
    "Hi, I'm Botty and we can chat if you want\nPlease type commands in lowercase English. Type quit to leave "
)  # default message at the start

# Starting the chatbot conversation
chat.converse()
