from django.shortcuts import render
from .models import Question

def index(request):
    question_list = Question.objects.order_by('-create_date')
    context = {'question_list': question_list}
    return render(request, 'pybo/question_list.html', context)
from .data3 import *

def test(request):
    context = {'result': get_player()}
    return render(request, 'pybo/question_list.html', context)