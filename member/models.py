from django.db import models

# Create your models here.

class Member(models.Model) :
    id = models.CharField(max_length=20, primary_key=True)
    pass1 = models.CharField(max_length=20)
    name = models.CharField(max_length=20)
    gender = models.IntegerField(default=0)
    tel = models.CharField(max_length=20)
    email = models.CharField(max_length=100)