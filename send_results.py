from secret import *


def sendEmail(contents, subject):
    import yagmail
    yag = yagmail.SMTP(gmail_adress, gmail_password)
    yag.send(email_recipient, subject, contents)
