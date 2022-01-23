#!/usr/bin/python
# -*- coding: UTF-8 -*-

import datetime
class TimeUnit:

  time=None
  time_type=None
  start_offset=None
  unit=None
  def date(self):
    return datetime.datetime(self.time.year,self.time.month,self.time.day)
  def convert_time(self,param_time,param_time_type,param_start_offset,param_unit):
    if param_time_type=="hour":
      time_to_return=datetime.datetime(param_time.year,param_time.month,param_time.day,param_time.hour)
      if (time_to_return.hour%param_unit)!=param_start_offset:
        delta_hour=(time_to_return.hour+param_start_offset)%param_unit
        time_to_return=time_to_return-datetime.timedelta(hours=delta_hour)
    if param_time_type=="week":
      time_to_return=datetime.datetime(param_time.year,param_time.month,param_time.day)
      if (time_to_return.hour%param_unit)!=param_start_offset:
        delta_hour=(time_to_return.hour+param_start_offset)%param_unit
        time_to_return=time_to_return-datetime.timedelta(hours=delta_hour)
    if param_time_type=="day":
      time_to_return=datetime.datetime(param_time.year,param_time.month,param_time.day)
    return time_to_return

  '''
  time: must be datetime format
  start_offset: offset hour/如果是day，则弃用
  unit: unit in hour or unit in 如果是day，则弃用
  time_type: need to be hour or 如果是day，则弃用
  '''
  def __init__(self,param_time,param_time_type,param_start_offset=2,param_unit=4):
    assert (param_time_type=="day")|(param_time_type=="hour")|(param_time_type=="week"), "time_type必须为hour或week的字符串" 
    assert param_time.__class__==datetime.datetime,"时间必须是datetime.datetime格式！"
    assert param_start_offset.__class__==int,"start_offset必须是int格式！"
    assert param_unit.__class__==int,"unit必须是int格式！"
    self.time=self.convert_time(param_time,param_time_type,param_start_offset,param_unit)
    self.time_type=param_time_type
    self.start_offset=param_start_offset
    self.unit=param_unit

  def __add__(self,another):
    raise Exception('TimeUnit这个类不可以相加！')

  def __sub__(self,another):
    assert self.time_type==another.time_type,"减法的时候，time_type必须一致！"
    assert self.start_offset==another.start_offset,"减法的时候，start_offset必须一致！"
    assert self.unit==another.unit,"减法的时候，unit必须一致！"
    if self.time_type=="hour":
      if self.time>another.time:
        return (self.time-another.time).total_seconds()/(3600*self.unit)
      else:
        return (another.time-self.time).total_seconds()/(3600*self.unit)
    if self.time_type=="day":
      if self.time>another.time:
        return (self.time-another.time).total_seconds()/(3600*24)
      else:
        return (another.time-self.time).total_seconds()/(3600*24)

  def start_time(self):
    return self.time

  def end_time(self):
    if self.time_type=="day":
      return self.time+datetime.timedelta(hours=24)
    return self.time+datetime.timedelta(hours=self.unit)

  def block_index(self):
    if self.time_type=="hour":
      if self.start_time().hour-2<0:
        return 24/unit
      return (self.start_time().hour-2)/4
    if self.time_type=="day":
      return self.start_time().weekday()

  #返回两个unit之间的天数的间隔，当type为day的时候，间隔是7天，当type为hour的时候，间隔是1天
  def time_interval_between_unit_in_day(self):
    if self.time_type=="day":
      return 7
    if self.time_type=="hour":
      return 1












