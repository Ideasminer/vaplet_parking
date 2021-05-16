from django.shortcuts import render
from django.http import HttpResponse, FileResponse, HttpResponseRedirect,StreamingHttpResponse, Http404
from vapsim_web.core.experiment import getsolution
from vapsim_web.core.ga_binary import *
from vapsim_web.core.simulation import *
from vapsim_web.core.layout_plot import draw
import os
from vapsim_web.models import *
from django.contrib import messages
from django.core.cache import cache, caches
from datetime import datetime,timedelta
import pytz

start_time = time.time()

# Create your views here.
def react(request):
    info = []
    status, login_name, login_pwd = login(request)[0], login(request)[1], login(request)[2]
    gamma, beta1, beta2, beta3 = getSetting(request, status, login_name)
    calculate(request, status, login_name, gamma, beta1, beta2, beta3)
    register(request)
    info = getHistory(status, login_name)
    info = [[i.width, i.height, i.policy, i.layout] for i in info]
    nickname = User.objects.filter(email=login_name)[0].name
    response = render(request,'index.html', {'info': info, 'email':login_name, 'name':nickname})
    if not login_name is None and not login_pwd is None and login_name !='None' and login_pwd != 'None':
        response.set_cookie(key='login_name', value = login_name, max_age=2000000)
        response.set_cookie(key='login_pwd', value = login_pwd, max_age=2000000)
    if not gamma is None and gamma != 'None' \
        and not beta1 is None and beta1 != 'None' \
        and not beta2 is None and beta2 != 'None' \
        and not beta3 is None and beta3 != 'None':
        response.set_cookie(key='gamma', value = gamma, max_age=2000000)
        response.set_cookie(key='beta1', value = beta1, max_age=2000000)
        response.set_cookie(key='beta2', value = beta2, max_age=2000000)
        response.set_cookie(key='beta3', value = beta3, max_age=2000000)
    return response


def login(request):
    # 初始默认状态为0，未登陆
    status = 0
    response = HttpResponse()
    # 尝试从COOKIES中提取登陆信息
    # 如果找不到，则需要用户自己登陆
    if request.POST and not request.POST.get('username') is None and not request.POST.get('login_password') is None:
        login_name = request.POST.get('username')
        login_pwd = request.POST.get('login_password')
    else:
        try:
            if request.COOKIES['login_name'] != 'None' and not request.COOKIES['login_name'] is None:
                login_name = request.COOKIES['login_name']
            else:
                login_name = None
            if request.COOKIES['login_pwd']  != 'None' and not request.COOKIES['login_pwd'] is None:
                login_pwd = request.COOKIES['login_pwd']
            else:
                login_pwd = None
        except:
            login_name = None
            login_pwd = None
    # 如果当前存在用户名和密码，则验证密码，进行登陆
    if not login_name is None and not login_pwd is None and login_name != 'None' and login_pwd != 'None':
        try:
            # 如果当前用户存在，则获得密码以及上次登陆和当前登陆的时间差
            target_pwd = User.objects.filter(email=login_name)[0].pwd
            pre_time = User.objects.filter(email=login_name)[0].login_time
            now = datetime.now()
            now = now.replace(tzinfo=pytz.timezone('UTC'))
            last_time = (now - pre_time).days
            # 如果超过1天没有登陆，则设置登陆状态值为0，即离线
            if last_time >= 1:
                User.objects.filter(email=login_name)[0].status = 0
                User.objects.filter(email=login_name)[0].is_active = True
                User.objects.filter(email=login_name)[0].save()
            # 如果密码不正确，提示重新登陆
            if target_pwd != login_pwd:
                messages.success(request,"密码错误！请重新输入")
            # 如果密码正确，则设置cookies，保存用户名和密码，并且读取当前状态值
            # 更新数据库中的状态值为1，登陆时间为当前登陆时间
            else:
                response.set_cookie(key='login_name', value = login_name, max_age=2000000)
                response.set_cookie(key='login_pwd', value = login_pwd, max_age=2000000)
                now = datetime.now()
                now = now.replace(tzinfo=pytz.timezone('UTC'))
                user = User.objects.filter(email=login_name)[0]
                user.login_time = now
                user.status = 1
                user.is_active = True
                user.save()
                # info为当前用户计算的所有历史信息
                info = History.objects.filter(email=login_name)
            # 此时进行了登陆，获取状态值
            user = User.objects.filter(email=login_name)[0]
            status = user.status
            return status, login_name, login_pwd
        except:
            # 当前用户不存在/用户名错误，状态值保持为初始值0，离线状态，并抛出alert
            return 0, None, None
    else:
        print(login_name, login_pwd)
        return 0, None, None

def register(request):
    if request.POST:
        name = request.POST.get('register_name')
        password = request.POST.get('password')
        email = request.POST.get('email')
    else:
        return 0
    # 如果注册请求列表均不为0，则进行注册
    if not name is None and not password is None and not email is None: 
        # 如果当前用户已经存在，则提醒已经被注册
        if len(User.objects.filter(email = email)) >= 1:
            messages.success(request,"该邮箱已被注册")
        else:
            # 否则，记录当前注册用户的昵称、密码和邮箱
            user = User(name=name, pwd=password, email = email)
            user.is_active = True
            user.save()

def calculate(request, status, login_name, gamma, beta1, beta2, beta3):
    if request.POST:
        # 获取参数
        width = request.POST.get('getwidth')
        height = request.POST.get('getheight')
        policy = request.POST.get('getpolicy')
        # 如果计算参数均存在，则进行计算
        if not width is None and not height is None and not policy is None:
            # 如果当前状态值为1，即在线状态，则进行计算
            if status == 1 or status == '1':
                width = int(eval(width))
                height = int(eval(height))
                bestsol = getsolution(width, height, policy)
                num_stack = int(np.floor((height - 3) / 2.5))
                draw(bestsol, num_stack)
                # 保存用户的用户名以及计算指标
                history = History(email = login_name, width=width, height=height, policy = policy, layout=bestsol)
                history.is_active = True
                history.save()
            else:
                # 如果当前状态值为0，但是提交了计算的请求，则抛出对话框，要求先登陆后使用,如果当前没有参数反馈，则说明没有提交计算请求，不进行计算
                messages.success(request,"请您登陆后使用")
    else:
        return 0


def getHistory(status, login_name):
    history = History.objects.filter(email=login_name)
    return history

def getSetting(request, status, login_name):
    if request.POST and not request.POST.get('gamma') is None \
        and not request.POST.get('beta1') is None \
        and not request.POST.get('beta2') is None \
        and not request.POST.get('beta3') is None:
        gamma = request.POST.get('gamma')
        beta1 = request.POST.get('beta1')
        beta2 = request.POST.get('beta2')
        beta3 = request.POST.get('beta3')
    else:
        try:
            gamma = request.COOKIES['gamma']
            beta1 = request.COOKIES['beta1']
            beta2 = request.COOKIES['beta2']
            beta3 = request.COOKIES['beta3']
        except:
            gamma, beta1, beta2, beta3 = None, None, None, None
    if status:
        if not gamma is None and gamma != 'None' \
            and not beta1 is None and beta1 != 'None' \
            and not beta2 is None and beta2 != 'None' \
            and not beta3 is None and beta3 != 'None':
            gamma = eval(gamma)
            beta1 = eval(beta1)
            beta2 = eval(beta2)
            beta3 = eval(beta3)
        else:
            gamma, beta1, beta2, beta3 = None, None, None, None
    else:
        gamma, beta1, beta2, beta3 = None, None, None, None
    return gamma, beta1, beta2, beta3

def download(request):
    with open('./vapsim_web/statics/images/layout.png', 'rb') as f:
        try:
            response = HttpResponse(f)
            response['content_type'] = "application/octet-stream"
            response['Content-Disposition'] = 'attachment; filename=layout.png'
            return response
        except Exception:
            raise Http404
    return response
