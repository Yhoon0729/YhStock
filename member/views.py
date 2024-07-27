import time

from django.contrib import auth
from django.http import HttpResponseRedirect
from django.shortcuts import render

from member.models import Member


# Create your views here.


def signup(request) :
    if request.method != "POST":
        return render(request, "member/signup.html")
    else :
        id1 = request.POST["id"]
        if Member.objects.filter(id=id1).exists():
            context = {"msg" : "존재하는 아이디 입니다.", "url" : "/member/signup/"}
            return render(request, "alert.html", context)
        else :
            member = Member(id=request.POST['id'],
                            pass1=request.POST['pass1'],
                            name=request.POST['name'],
                            gender=request.POST['gender'],
                            tel=request.POST['tel'],
                            email=request.POST['email'])
            member.save()
            context = {"msg" : "회원가입을 환영합니다.", "url" : "/member/login/"}
            return render(request, "alert.html", context)

def login(request):
    if request.method != "POST":
        return render(request, "member/login.html")
    else :
        id1 = request.POST["id"]
        pass1 = request.POST["pass1"]
        try :
            member = Member.objects.get(id=id1)
        except :
            context = {"msg" : "아이디를 확인하세요", "url" : "/member/login/"}
            return render(request, "alert.html", context)
        else :
            if pass1 == member.pass1:
                request.session['id'] = id1
                time.sleep(1)
                context = {"msg" : "환영합니다.", "url" : "/stock/index/"}
                return render(request, "alert.html", context)

            else :
                context = {"msg" : "비밀번호를 확인하세요.", "url":"/member/login/"}
                return render(request, "alert.html", context)

def logout(request) :
    auth.logout(request)
    context = {"msg" : "로그아웃 되었습니다.", "url" : "/member/login"}
    return render(request, "alert.html", context)