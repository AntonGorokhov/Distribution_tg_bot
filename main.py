from aiogram import Bot, Dispatcher, executor, types
from aiogram.types import ReplyKeyboardRemove, ReplyKeyboardMarkup, KeyboardButton
from aiogram.dispatcher.filters.state import State, StatesGroup
from aiogram.contrib.fsm_storage.memory import MemoryStorage
from aiogram.dispatcher.filters import Text
from aiogram.dispatcher import FSMContext

API_TOKEN = '5590116409:AAEVO54ur3KDSr_QEsR6Cl82urqmu3SkakA'

bot = Bot(token=API_TOKEN)

dp = Dispatcher(bot, storage=MemoryStorage())


# Reminder, Calculator

# Distributions: Norm, Exp, Pois -> 
# Calculation: from A to B P(A <= x <= B) for each distribution

#answer is always no


kbActions = [
        [
            types.KeyboardButton(text="Get PDF"),
            types.KeyboardButton(text="Calculate Probability"),
            types.KeyboardButton(text="Cancel")
        ],
    ]

kbDistributions = [
        [
            types.KeyboardButton(text="Normal"),
            types.KeyboardButton(text="Pois"),
            types.KeyboardButton(text="Exp"),
            types.KeyboardButton(text="Cancel")
        ],
    ]



class Form(StatesGroup):
    action = State() 
    calc = State()
    pdf = State()
    seg_norm = State()
    seg_pois = State()
    seg_exp = State()



# start
@dp.message_handler(commands=['start', 'help'])
async def send_welcome(message: types.Message):
    kb = kbActions
    keyboard = types.ReplyKeyboardMarkup(keyboard=kb)
    await Form.action.set()
    await message.reply("Welcome to distribution bot, please select an action", reply_markup=keyboard)


#cancel
@dp.message_handler(state='*', commands='cancel')
@dp.message_handler(Text(equals='cancel', ignore_case=True), state='*')
async def cancel_handler(message: types.Message, state: FSMContext):
    current_state = await state.get_state()
    if current_state is None:
        return

    # Cancel state and inform user about it
    await state.finish()
    # And remove keyboard (just in case)
    await message.reply('Cancelled. Type /start to begin', reply_markup=types.ReplyKeyboardRemove())


#action
@dp.message_handler(state=Form.action)
async def select_distribution(message: types.Message):
    if message.text == "Get PDF":
        await Form.pdf.set()
    elif message.text == "Calculate Probability":
        await Form.calc.set()
    else:
        await message.reply("Wrong action, please retry")
        await Form.action.set()
        return

    kb = kbDistributions
    keyboard = types.ReplyKeyboardMarkup(keyboard=kb)
    await message.reply("Select distribution", reply_markup=keyboard)



#distribution
@dp.message_handler(state=Form.pdf)
async def select_distribution(message: types.Message):
    if message.text == "Pois":
        await message.reply("Lambda: p(x = k) = Lambda^k / k! * e^(-Lambda)")
        return
    elif message.text == "Normal":
        await message.reply("p(x) = 1 / sqrt(2pi) * e^(-x^2 / 2)")
        return
    elif message.text == "Exp":
        await message.reply("Lambda: p(x) = Lambda * e^(-x * Lambda)")
        return
    else:
        await message.reply("Wrong distribution selected, please retry")
        return


@dp.message_handler(state=Form.calc)
async def select_distribution(message: types.Message):
    if message.text == "Pois":
        await Form.seg_pois.set()
        await message.reply("Type 2 integers for Pois distribution")
        return
    elif message.text == "Normal":
        await Form.seg_norm.set()
        await message.reply("Type 2 integers for Normal distribution")
        return
    elif message.text == "Exp":
        await Form.seg_exp.set()
        await message.reply("Type 2 integers for Exp distribution")
        return
    else:
        await message.reply("Wrong distribution selected, please retry")
        return

def check_int(s):
    if s[0] in ('-', '+'):
        return s[1:].isdigit()
    return s.isdigit()



@dp.message_handler(state=Form.seg_norm)
async def select_distribution(message: types.Message):
    lst = message.text.split()
    if len(lst) != 2:
        await message.reply("Segment must contain only 2 integers")
        return
    if not (check_int(lst[0]) and check_int(lst[1])):
        await message.reply("Segment must contain only 2 INTEGERS")
        return
    await message.reply("No")


@dp.message_handler(state=Form.seg_pois)
async def select_distribution(message: types.Message):
    lst = message.text.split()
    if len(lst) != 2:
        await message.reply("Segment must contain only 2 integers")
        return
    if not (check_int(lst[0]) and check_int(lst[1])):
        await message.reply("Segment must contain only 2 INTEGERS")
        return
    await message.reply("No")


@dp.message_handler(state=Form.seg_exp)
async def select_distribution(message: types.Message):
    lst = message.text.split()
    if len(lst) != 2:
        await message.reply("Segment must contain only 2 integers")
        return
    if not (check_int(lst[0]) and check_int(lst[1])):
        await message.reply("Segment must contain only 2 INTEGERS")
        return
    await message.reply("No")
    
    
    








if __name__ == '__main__':
    executor.start_polling(dp, skip_updates=True)


