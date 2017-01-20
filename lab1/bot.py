from sopel import module
from emo.wdemotions import EmotionDetector

emo = EmotionDetector()

triggerlist =[]
countlist=[]


@module.rule('')
def hi(bot, trigger):
    print(trigger, trigger.nick)
	
    bot.say('Hi, ' + trigger.nick)
	
	count = emo.detect_emotuion_in_row(str(trigger))
	triggerlist.add(trigger)
	countlist.add(count)
	average=count/triggerlist.size()
	print(average)

	
	
	
