from django.db import models
import datetime

# Create your models here.
class User(models.Model):
    name = models.CharField(max_length=30)
    pwd = models.CharField(max_length=30)
    email = models.CharField(max_length=30)
    status = models.IntegerField(default=0)
    login_time = models.DateTimeField(default=datetime.datetime.now() - datetime.timedelta(1000))

class History(models.Model):
    email = models.CharField(max_length=30)
    width = models.IntegerField(blank=True)
    height = models.IntegerField(blank=True)
    policy = models.CharField(max_length=10)
    layout = models.CharField(max_length=50)