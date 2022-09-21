---
layout: page
title: codes
---

Preparation for coding tests  
references:  
* [문제 리스트](https://covenant.tistory.com/224)  
* [파이썬 sys 입력 받기](https://velog.io/@yeseolee/Python-%ED%8C%8C%EC%9D%B4%EC%8D%AC-%EC%9E%85%EB%A0%A5-%EC%A0%95%EB%A6%ACsys.stdin.readline)
* [최단경로 알고리즘-다익스트라,벨만포드,플로이드](https://velog.io/@ehdrms2034/%EC%95%8C%EA%B3%A0%EB%A6%AC%EC%A6%98-%EC%B5%9C%EB%8B%A8-%EA%B2%BD%EB%A1%9C-%EC%95%8C%EA%B3%A0%EB%A6%AC%EC%A6%98-%EB%8B%A4%EC%9D%B5%EC%8A%A4%ED%8A%B8%EB%9D%BC-%EB%B2%A8%EB%A7%8C-%ED%8F%AC%EB%93%9C-%ED%94%8C%EB%A1%9C%EC%9D%B4%EB%93%9C-%EC%95%8C%EA%B3%A0%EB%A6%AC%EC%A6%98)
* [Priority queue](https://velog.io/@mein-figur/Python%EC%9A%B0%EC%84%A0%EC%88%9C%EC%9C%84-%ED%81%90-heapq)
* [MST-크루스칼,프림](https://velog.io/@ready2start/Python-MST-%EC%95%8C%EA%B3%A0%EB%A6%AC%EC%A6%98)


<br>

### Difficult problems

{% for post in site.posts %}
  {% if post.tags contains "codes" %}
    {% if post.star %}
  * {{ post.level }} &raquo; [ {{ post.title }} ]({{ post.url }})
    {% endif %}
  {% endif %}
{% endfor %}

### Platinum

{% for post in site.posts %}
  {% if post.tags contains "codes" %}
    {% if post.level contains "platinum" %}
  * {{ post.level }} &raquo; [ {{ post.title }} ]({{ post.url }})
    {% endif %}
  {% endif %}
{% endfor %}

### Gold

{% for post in site.posts %}
  {% if post.tags contains "codes" %}
    {% if post.level contains "gold" %}
  * {{ post.level }} &raquo; [ {{ post.title }} ]({{ post.url }})
    {% endif %}
  {% endif %}
{% endfor %}

### Silver

{% for post in site.posts %}
  {% if post.tags contains "codes" %}
    {% if post.level contains "silver" %}
  * {{ post.level }} &raquo; [ {{ post.title }} ]({{ post.url }})
    {% endif %}
  {% endif %}
{% endfor %}

### Bronze

{% for post in site.posts %}
  {% if post.tags contains "codes" %}
    {% if post.level contains "bronze" %}
  * {{ post.level }} &raquo; [ {{ post.title }} ]({{ post.url }})
    {% endif %}
  {% endif %}
{% endfor %}