## happy path 1 (/var/folders/g8/jg5c6c5x65l7q2glgx33_8xw0000gn/T/tmp82zuhc3y/e538151e6e354f6db79f0b5da1986e18_conversation_tests.md)
* greet: hello there!
    - utter_greet   <!-- predicted: utter_botgreet -->
* mood_great: amazing
    - utter_happy   <!-- predicted: utter_thanks -->


## happy path 2 (/var/folders/g8/jg5c6c5x65l7q2glgx33_8xw0000gn/T/tmp82zuhc3y/e538151e6e354f6db79f0b5da1986e18_conversation_tests.md)
* greet: hello there!
    - utter_greet   <!-- predicted: utter_botgreet -->
* mood_great: amazing
    - utter_happy   <!-- predicted: utter_thanks -->
* goodbye: bye-bye!
    - utter_goodbye   <!-- predicted: utter_botgreet -->


## sad path 1 (/var/folders/g8/jg5c6c5x65l7q2glgx33_8xw0000gn/T/tmp82zuhc3y/e538151e6e354f6db79f0b5da1986e18_conversation_tests.md)
* greet: hello
    - utter_greet   <!-- predicted: utter_botgreet -->
* mood_unhappy: not good
    - utter_cheer_up   <!-- predicted: utter_veg_non_veg_egg -->
    - utter_did_that_help   <!-- predicted: action_listen -->
* affirm: yes
    - utter_happy   <!-- predicted: utter_anthingelse -->


## sad path 2 (/var/folders/g8/jg5c6c5x65l7q2glgx33_8xw0000gn/T/tmp82zuhc3y/e538151e6e354f6db79f0b5da1986e18_conversation_tests.md)
* greet: hello
    - utter_greet   <!-- predicted: utter_botgreet -->
* mood_unhappy: not good
    - utter_cheer_up   <!-- predicted: utter_veg_non_veg_egg -->
    - utter_did_that_help   <!-- predicted: action_listen -->
* deny: not really
    - utter_goodbye   <!-- predicted: action_hello_world -->


## sad path 3 (/var/folders/g8/jg5c6c5x65l7q2glgx33_8xw0000gn/T/tmp82zuhc3y/e538151e6e354f6db79f0b5da1986e18_conversation_tests.md)
* greet: hi
    - utter_greet   <!-- predicted: utter_botgreet -->
* mood_unhappy: very terrible
    - utter_cheer_up   <!-- predicted: utter_veg_non_veg_egg -->
    - utter_did_that_help   <!-- predicted: action_listen -->
* deny: no
    - utter_goodbye   <!-- predicted: action_hello_world -->


## say goodbye (/var/folders/g8/jg5c6c5x65l7q2glgx33_8xw0000gn/T/tmp82zuhc3y/e538151e6e354f6db79f0b5da1986e18_conversation_tests.md)
* goodbye: bye-bye!
    - utter_goodbye   <!-- predicted: utter_anthingelse -->


## bot challenge (/var/folders/g8/jg5c6c5x65l7q2glgx33_8xw0000gn/T/tmp82zuhc3y/e538151e6e354f6db79f0b5da1986e18_conversation_tests.md)
* bot_challenge: are you a bot?
    - utter_iamabot   <!-- predicted: utter_thanks -->


