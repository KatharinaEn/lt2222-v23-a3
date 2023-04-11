# LT2222 V23 Assignment 3


Part 4 - Enron data ethics:

Since the messages were written during office hours from office computers, and were therefore the property of the company, I would say, it is understandably, that these messages will be used against these people later on in court since it's the property of the company.

On the other hand, messages are always a private issue - even though they are written at an office computer (and on the office server). Therefore it is difficult, to use them in public trials or similar public affairs. In addition, the email history was made public without the knowdledge of the people. With having made them public people were arrested because of the statements they have written in these emails.

All in all, ethics was and became, I would say, even more an important topic. Using such chat/email histories for machine training purposes, and in a university context is acceptable I would say, because explicitly these mails are already published and public. However, using these mails in another context, such as, for a campain or other public event, is not quite acceptable since then the permission of the authors would be needed (even though these mails are public to everyone). 

Part 5 - documentation:


Please run the file for part 1 as follows: python3 a3_features.py ./data/enron_sample/ data.pickle 400 --test=20

In part 1 I extracted the names of the authors, which I called Y. As well as I extracted the headers and signature lines from the emails, resulting in creating "mailtext" which equals my X. I used X (mailtext) to create the vectors. Then I used the outcome of the vectors, which I called matrix to create the train_test_split of 80/20 with sklearn method train_test_split. I then put the output of x_train, y_train, x_test, y_test into a list called "data". Finally, I used pickle.dumb to write it the data out to the args.outputfile.

Part 2: 

I started on part 2, however, my code doesn't work, it throws errors.
I still submitted it, because I've written it, and put quite much time in it.
However, part 2 can be run with the following command: python3 a3_model.py data.pickle
