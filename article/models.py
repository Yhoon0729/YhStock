from django.db import models

# Create your models here.
class Article(models.Model):
    # num 값이 없으면 자동증가함. (auto increments)
    num = models.AutoField(primary_key=True)
    name = models.CharField(max_length=30)
    pass1 = models.CharField(max_length=20) # pass는 예약어
    subject = models.CharField(max_length=100)
    content = models.CharField(max_length=4000)
    regdate = models.DateField(auto_now_add=True, blank=True) # 오늘 날짜
    readcnt = models.IntegerField(default=0)
    file1 = models.CharField(max_length=300)

    def __str__(self) :
        return str(self.num) + ":" + self.subject

class Comment(models.Model) :
    # ser 값이 없으면 자동증가함 (auto incrementes)
    ser = models.AutoField(primary_key=True)
    num = models.IntegerField(default=0)
    content = models.CharField(max_length=4000)
    regdate = models.DateField(auto_now_add=True, blank=True)

    def __str__(self) :
        return str(self.ser) + "\t" + ":" + "\t" + self.content