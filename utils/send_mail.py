import smtplib
def send_mail(message):
    try:
        server = smtplib.SMTP('smtp.ym.163.com','25')  
        server.login("microsoft@huangqiushi.com","microsoft")  
        msg = message
        server.sendmail("microsoft@huangqiushi.com", "hqsiswiliamdev@163.com", msg)  
    except Exception, e:
        pass
