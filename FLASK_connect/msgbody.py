#!/usr/bin/env python
# coding: utf-8

# In[13]:


import numpy as np
import pandas as pd
import random
from random import randint


# In[14]:


currency = ['USD','GBP','INR','EUR']


# In[15]:


def numgen(n):
    return ''.join(["{}".format(randint(1, 9)) for num in range(0, n)])
    #range_start = 10**(n-1)
    #range_end = (10**n)-1
    #return randint(range_start, range_end)


# In[16]:


def subjectGenerator(class_,ID):
    temp = {
        'Complete' : [
            'Transaction ' + ID + ' is complete',
            'Completed transaction no. ' + ID ,
            'Payment done and Transaction ' + ID + ' settled.',
            'Finalized transaction of ID : ' + ID,
            'Sorted out the transaction with ID : ' + ID,
            'Concluded the transaction ' + ID,
            'Fulfilled transaction having ID : ' + ID,
            'Received full payment for transaction no. ' + ID
        ],
        'Failed' : [
            'Transaction ' + ID + ' has failed!',
            'Failure of transaction ' + ID,
            'Urgent : Transaction ' + ID + ' ceased.',
            'Transaction ' + ID + ' stalled and payment not received.',
            'Transaction having ID : ' + ID + ' has stopped. Help!',
            'Abrupt closure of transaction with ID ' + ID,
            'Why has my transaction ' + ID + ' stopped?'
        ],
        'Pending' : [
            'Payment is pending for transaction ' + ID,
            'Pending payment for transaction having ID : ' + ID,
            'Partial payment for transaction ' + ID,
            'Partially paid the required amount for transaction ' + ID,
            'The pending amount for transaction ' + ID + ' will reach you soon',
            'Incomplete transaction ' + ID,
            'Transaction no. ' + ID + ' is unresolved.',
            'Payment outstanding for transaction ' + ID,
            'Remaining amount for transaction ' + ID + ' to be paid later.'
        ],
        'Request' : [
            'Request to send details of transaction ' + ID,
            'Seeking update on the status of transaction ' + ID,
            'Asking for the details for transaction ' + ID,
            'Soliciting information for ID : ' + ID,
            'Urgently required update on transaction ' + ID,
            'Impetrating details for ID : ' + ID,
            'Imploring update on transaction : ' + ID
        ],
        'Processing' : [
            'Payment received for transaction : ' + ID + ' and now processing. ',
            'Handling the transaction ' + ID + ' after payment.',
            'Transaction ' + ID + ' is now being processed.',
            'Required money acquired. Transaction ' + ID + ' is in process.',
            'Accepted payment. Transaction' + ID + ' currently processing',
            'Dealing with the transaction ' + ID
        ],
        'General' : [
            'Send steps to activate online banking',
            'How to change PIN no of ATM card?',
            'Want to block account ' + ID,
            'New Cheque book',
            'Add one more contact no. to account ' + ID,
            'Change address for account no. ' + ID,
            'Upgrade to an account with more benefits',
            'Why cant I withdraw money with my ATM card?'
        ]
    }
    return random.choice(temp[class_])


# In[23]:


def bodyGenerator(class_,ID):
    temp = {
        'Complete' : [
            'Hello! Sincere greetings for the day. I would like to inform you that my transaction ' + ID + ' has completed. Thank you so much for your support. Looking forward to working more with you in the future. Regards',
            'To whom it may concern, I have successfully received payment for the transaction ' + ID + ' . I am grateful for your cooperation. Thank you so much and regards.',
            'Hey, I am writing in reference to the transaction ' + ID + ' . I was granted the aforementioned amount within the deadline. Sincere gratitude for such a quick response. Kind regards',
            'Glad to let you know that I got the desired payment in reference to transaction ' + ID + ' .I am very happy with your services and will definitely recommend your company to my friends and acquaintances. Warm regards.',
            'Greetings! I wanted to let you know that I have acknowledged the payment for transaction ' + ID + ' in response to your email which confirmed the finalized status of my transaction. Thanks and regards.',
            'I deeply appreciate your quick service as I have received the pre-approved loan amount of ' + random.choice(currency) + ' ' + numgen(6)
        ],
        'Failed' : [
            'This is in response to your email stating that my transaction having ID ' + ID + ' has failed but no reason was mentioned. Can you please tell me what did I do wrong so that I can create a new one without any errors? Waiting for your reply. Thank you in advance.',
            'Greetings for the day, I checked my inbox and found your email stating the failure of my transaction. Please help me understand why ' + ID + ' has failed, need funds urgently! Patiently waiting for you reply. Thanks a lot.',
            'Hey, I see my transaction with ID ' + ID + ' has failed. I think I did everything right? Please look into this and reply with the reason urgently. Thank you and regards.',
            'This is to notify you that my transaction ' + ID + ' has failed. Please reply to me with the cause as soon as possible.',
            'I have been your regular client and have followed the procedure for creating a transaction correctly. Still I received an email saying that the transaction has failed. I would like to know the reasoning. My Transaction ID is ' + ID + ' Regards.'
        ],
        'Pending' : [
            'I regret to inform you the I could only pay the partial amount of ' + random.choice(currency) + ' ' + numgen(6) + ' I will definitely pay the remaining amount as soon as possible.Thank you for understanding. Regards',
            'There has been only a partial payment of amount ' + random.choice(currency) + ' ' + numgen(6) + '.Assuring you that the rest will be paid later. Warm Regards.',
            'Since my transaction ' + ID + ' is still pending, I wanted to know if there is a problem with the paperwork from my side. Please let me know at the earliest. Thanks and regards.',
            'The transaction ' + ID + ' is taking too long to complete. I would request you to kindly guide me through the further steps to be taken in order to complete the transaction.',
            'Hello!! Greetings for the day. Status of transaction ' + random.choice(currency) + ' ' + numgen(6) + ' for account ' + ID + ' is pending, I would be grateful if you could tell me the cause. Thanks a lot.'
        ],
        'Request' : [
             'Please provide a status update on ' + ID + ' at the earliest',
             'I need details of ' + ID + ' urgently, please provide the same on priority basis.',
             'Urgently require details of acc ' + ID + ' reply asap',
             'Hey, I would like to know the details of account no. ' + ID + 'Thanks.',
             'I request you to kindly send the status of my transaction with ID : '+ ID + 'Thanks and regards',
             'Kindly reply to me at the earliest with the last 3 transactions made with the account no ' + ID + 'Thanks in advance.',
             'Can you please tell me the amount transferred through transaction ID : ' + ID + '? Thanks.'
        ],
        'Processing' : [
            'Hello! This is to inform you that I have received the amount you transferred to my account and now it is currently in process.'
        ],
        'General' : [
            'Hey, Can you please send me the detailed steps that I ought to follow to create a new bank account?',
            'I have heard a lot about your bank and would like to open an account with CitiBank. For that, I would like to know all the documents that are required. Kindly reply with the same and also the further steps to be taken. Eagerly waiting for your reply.',
            'Hey, I think I lost my ATM card today so can you please block my account or tell me a way to change my PIN no. ? I urgently need your reply . Thanks in advance.'
        ]
    }
    return random.choice(temp[class_])


# In[ ]:




