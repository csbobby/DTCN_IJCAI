import sqlite3
import pdb

def judge_finished(model_path):
  log_db = sqlite3.connect('running_logs.db')
  cursor=log_db.cursor()
  cursor.execute('''CREATE TABLE if not exists log (model_path text, epoch integer, finished text, losses text, val_losses text)''')
  cursor.execute("select model_path, epoch, finished from log where model_path='%s'"%model_path)
  row=cursor.fetchone()
  if row==None:
    log_db.close()
    return False
  elif row[2]=="False":
    log_db.close()
    return False
  elif row[2]=="True":
    log_db.close()
    return True
  return False

def get_epoch_num(model_path):
  log_db = sqlite3.connect('running_logs.db')
  cursor=log_db.cursor()
  cursor.execute('''CREATE TABLE if not exists log (model_path text, epoch integer, finished text, losses text, val_losses text)''')
  cursor.execute("select model_path, epoch, finished from log where model_path='%s'"%model_path)
  row=cursor.fetchone()
  if row==None:
    print "Record not exist!"
    log_db.close()
    return 0
  else:
    log_db.close()
    return row[1]

def update_status(model_path,epoch,finished):
  log_db = sqlite3.connect('running_logs.db')
  cursor=log_db.cursor()
  cursor.execute('''CREATE TABLE if not exists log (model_path text, epoch integer, finished text, losses text, val_losses text)''')
  cursor.execute("select model_path, epoch, finished from log where model_path='%s'"%model_path)
  if None==cursor.fetchone():
    cursor.execute("insert into log (model_path,epoch,finished) VALUES('%s',%d,'%s')"%(model_path,epoch,finished))
    log_db.commit()
    log_db.close()
    return True
  else:
    cursor.execute("update log set epoch=%d,finished='%s' where model_path='%s'"%(epoch,finished,model_path))
    log_db.commit()
    log_db.close()
    return True

def save_loss(model_path,losses):
  log_db = sqlite3.connect('running_logs.db')
  cursor=log_db.cursor()
  serialize_str=', '.join([str(x) for x in losses])
  cursor.execute("update log set losses='%s' where model_path='%s'"%(serialize_str,model_path))
  log_db.commit()
  log_db.close()


def get_loss(model_path):
  log_db = sqlite3.connect('running_logs.db')
  cursor=log_db.cursor()
  cursor.execute("select losses from log where model_path='%s'"%(model_path))
  serialize_str=cursor.fetchone()[0]
  result = eval('[' + serialize_str + ']')
  return result


def save_val_loss(model_path,val_losses):
  log_db = sqlite3.connect('running_logs.db')
  cursor=log_db.cursor()
  serialize_str=', '.join([str(x) for x in val_losses])
  cursor.execute("update log set val_losses='%s' where model_path='%s'"%(serialize_str,model_path))
  log_db.commit()
  log_db.close()


def get_val_loss(model_path):
  log_db = sqlite3.connect('running_logs.db')
  cursor=log_db.cursor()
  cursor.execute("select val_losses from log where model_path='%s'"%(model_path))
  serialize_str=cursor.fetchone()[0]
  result = eval('[' + serialize_str + ']')
  return result