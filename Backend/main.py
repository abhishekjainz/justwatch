import logging
import random
from telegram import ReplyKeyboardMarkup, ReplyKeyboardRemove, Update, ParseMode
from telegram.ext import (
    Updater,
    CommandHandler,
    MessageHandler,
    Filters,
    ConversationHandler,
    CallbackContext,
)
from model import recommend_movie
from model import random_recommendation
from model import genre_recommendation
from model import recommend_movie_backend

from config import *

from faunadb import query as q
from faunadb.client import FaunaClient
from faunadb.errors import NotFound

# Fauna Client Config
client = FaunaClient(secret=FAUNA_KEY)

# Enable Logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO
)
logger = logging.getLogger(__name__)

PORT = int(os.environ.get('PORT', '8443'))

COMPANY, DEGREE, COLOUR, MOOD, FINAL, FINAL_2 = range(6)
GENRE_2 = range(1)

def start(update, context: CallbackContext) -> int:
    ''' Starts the conversation by introducing itself '''
    bot = context.bot
    chat_id = update.message.from_user.username
    if client.query(q.exists(q.match(q.index("userID"), chat_id))):
        update.message.reply_text(
            "Hi there! Welcome back to JustWatch!"
            "Please select your choice of command to kickstart this process :)"
    )
    else:
        update.message.reply_text(
            "Hi there! This seems like your first time here. Press /help to learn how this bot works!"
        )

def help(update, context):
    ''' Lists all the commands '''
    bot = context.bot
    chat_id = update.message.from_user.username
    update.message.reply_text(
        "Commands: \n" + "\n" +

        "/start: Activates the bot \n" + "\n"
        "/random: Recommends a random movie \n" + "\n"
        "/genre: Recommends a random movie from the desired genre \n" + "\n"
        "/personalized: Recommends a personalized movie \n" + "\n"
        "/website: Returns a link that connects to our website \n" + "\n"
        "/cancel: Deactivates the bot"
    )

def website(update, context):
    bot = context.bot
    chat_id = update.message.from_user.username
    bot.send_message(chat_id=update.message.chat_id, 
    text="<a href='https://justwatch01.herokuapp.com/'>JustWatch Website</a>", parse_mode=ParseMode.HTML)

###### RANDOM RECOMMENDATION ######
def random_rec(update: Update, _: CallbackContext) -> int:
    ''' Recommends a completely random movie '''
    update.message.reply_text(
        "Here is a random movie recommendation for you: " + random_recommendation()
    )

###### GENRE RECOMMENDATION ###### 
def genre(update: Update, _: CallbackContext) -> int:
    ''' Asks User to select a specific Genre'''
    user = update.message.from_user
    update.message.reply_text(
        "From which Genre would you like to watch a movie?"
    )
    return GENRE_2

def genre_2(update: Update, _: CallbackContext) -> int:
    ''' Recommends a random movie from a selected Genre'''
    user = update.message.from_user
    logger.info("%s wants to watch a movie from Genre: %s", user.first_name, update.message.text)
    update.message.reply_text(
        "Here is a movie for you from " +  update.message.text + ": " +
         genre_recommendation(update.message.text)
    )
    return ConversationHandler.END

###### PERSONALISED RECOMMENDATION ###### 
def personalized(update, context):
    ''' Starts the recommendation thread '''
    bot = context.bot
    chat_id = update.message.from_user.username

    if not client.query(q.exists(q.match(q.index("userID"), chat_id))):
        update.message.reply_text(
            "To recommend you the perfect movie, I would need you to answer few questions for me! "
            "Since, this is your first time, I would like to know your favourite 3 movies. "
            "Please type them in separated by a comma. For example; Matrix, Godzilla, Meg." 
        )
    else:
        update.message.reply_text(
        "Since we have already recorded your most favourite movies, we can skip that step. "
        "Please type 'next' to go to the next question! "
        ) 
    return COMPANY

def company(update, context):
    ''' Asks the user who would they be watching the movie with? '''

    bot = context.bot
    chat_id = update.message.from_user.username

    data = update.message.text.split(',')
    
    # Check if user already exists before creating new users
    if not client.query(q.exists(q.match(q.index("userID"), chat_id))):
        new_user = client.query(
            q.create(q.collection('Users'), {
                "data": {
                    "userID": chat_id,
                    "movie": data
                }
            })
        )
        context.user_data["userID"] = new_user["ref"].id()
        context.user_data["movie"] = new_user['data']

    # Asking next question
    reply_keyboard = [['Alone', 'Friends', 'Family', 'S/O']]

    if client.query(q.exists(q.match(q.index("userID"), chat_id))):
        update.message.reply_text(
        "Next question, Who are you intending to watch the movie with?",
        reply_markup = ReplyKeyboardMarkup(reply_keyboard, one_time_keyboard=True),
        )
    return DEGREE

def degree(update: Update, _: CallbackContext) -> int:
    ''' Asks the user whether they would prefer a light or a heavy movie?'''
    reply_keyboard = [['Light', 'Heavy', 'No Preference']]
    user = update.message.from_user
    logger.info("%s is currently watching with: %s", user.first_name, update.message.text)
    update.message.reply_text(
        'Would you prefer to watch a Light or a Heavy movie?',
        reply_markup = ReplyKeyboardMarkup(reply_keyboard, one_time_keyboard=True),
    )
    return COLOUR

def colour(update: Update, _: CallbackContext) -> int:
    ''' Asks the user for their favourite colour'''
    reply_keyboard = [['Red', 'Blue', 'Black', 'Green', 'White', 'None']]
    user = update.message.from_user
    logger.info("%s prefers watching a %s degree movie" , user.first_name, update.message.text)
    update.message.reply_text(
        'Which colour do you prefer?',
        reply_markup = ReplyKeyboardMarkup(reply_keyboard, one_time_keyboard=True),
    )
    return MOOD

def mood(update: Update, _: CallbackContext) -> int:
    ''' Asks the user for their current mood'''
    reply_keyboard =  [['Excited', 'Lonely', 'Depressed', 'Cheerful', 'Stressed']]
    user = update.message.from_user
    logger.info("The favourite colour of %s is %s", user.first_name, update.message.text)
    update.message.reply_text(
        'Which word best describes your feeling now?',
        reply_markup = ReplyKeyboardMarkup(reply_keyboard, one_time_keyboard=True),
    )

    return FINAL
    
    # return ConversationHandler.END

def final(update: Update, _: CallbackContext) -> int:
    ''' recommends movie to the User '''
    user = update.message.from_user
    chat_id = update.message.from_user.username
    logger.info("%s is currently feeling: %s", user.first_name, update.message.text)

    result = client.query(
    q.get(q.match(q.index("userID"), chat_id)))
    output = result['data']['movie']
    final_output = random.choice(output)

    try:
        recommend_movie_backend(final_output)
    except:
        if update.message.text == 'Excited':
            update.message.reply_text(
            "We recommend you: " + recommend_movie('Excited') + "!" + 
            " Are you happy with this recommendation?" + " y or n"
            )
        elif update.message.text == 'Lonely':
            update.message.reply_text(
                "We recommend you: " + recommend_movie('Lonely') + "!" + 
                " Are you happy with this recommendation?" + " y or n"
            )
        elif update.message.text == 'Depressed':
            update.message.reply_text(
                "We recommend you: " + recommend_movie('Depressed') + "!" +
                " Are you happy with this recommendation?" + " y or n"
            )
        elif update.message.text == 'Cheerful':
            update.message.reply_text(
                "We recommend you: " + recommend_movie('Cheerful') + "!" +
                " Are you happy with this recommendation?" + " y or n"
            )
        elif update.message.text == 'Stressed':
            update.message.reply_text(
                "We recommend you: " + recommend_movie('Stressed') + "!" + 
                " Are you happy with this recommendation?" + " y or n"
            )

    return FINAL_2

def final_2(update: Update, _: CallbackContext) -> int:
    ''' Thanking Note '''
    user = update.message.from_user
    chat_id = update.message.from_user.username
    logger.info("Did %s like the movie? : %s", user.first_name, update.message.text)

    update.message.reply_text(
            "Thank you for your feedback! This will allow us to constantly retrain our model for better results! " +
            " You can always get another recommendation by clicking on /cancel and then /start to reactive the bot!"
        )
        
    return ConversationHandler.END
    
def cancel(update: Update, _: CallbackContext) -> int:
    ''' Cancels and ends the conversation '''
    user = update.message.from_user
    logger.info("User %s canceled the conversation.", user.first_name)
    update.message.reply_text(
        'Bye! I hope I can help you again some day :)', reply_markup=ReplyKeyboardRemove()
        )

    return ConversationHandler.END

def main() -> None:
    ''' Run the bot '''
    updater = Updater(TOKEN)
    dispatcher = updater.dispatcher
    dispatcher.add_handler(CommandHandler("start", start))
    dispatcher.add_handler(CommandHandler("random", random_rec))
    dispatcher.add_handler(CommandHandler("help", help))
    dispatcher.add_handler(CommandHandler("cancel", cancel))
    dispatcher.add_handler(CommandHandler("website", website))

    # Add conversation handler with the states
    conv_handler = ConversationHandler(
        entry_points=[CommandHandler('personalized', personalized)],
        states={
            COMPANY: [MessageHandler(Filters.text & ~Filters.command, company)],
            DEGREE: [MessageHandler(Filters.text & ~Filters.command, degree)],
            COLOUR: [MessageHandler(Filters.text & ~Filters.command, colour)],
            MOOD: [MessageHandler(Filters.text & ~Filters.command, mood)],
            FINAL : [MessageHandler(Filters.text & ~Filters.command, final)],
            FINAL_2 : [MessageHandler(Filters.text & ~Filters.command, final_2)]
        },
        fallbacks=[CommandHandler('cancel', cancel)],
    )

    conv_handler_2 = ConversationHandler(
        entry_points=[CommandHandler('genre', genre)],
        states={
            GENRE_2: [MessageHandler(Filters.text & ~Filters.command, genre_2)],
        },
        fallbacks=[CommandHandler('cancel', cancel)],
    )

    dispatcher.add_handler(conv_handler)
    dispatcher.add_handler(conv_handler_2)

    updater.start_polling()
    updater.start_webhook(listen="0.0.0.0",
                      port=int(PORT),
                      url_path=TOKEN,
                      webhook_url='https://testingml.herokuapp.com/' + TOKEN)

    updater.idle()

if __name__ == '__main__':
    main()















