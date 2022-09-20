---
layout: page
title: codes
---

Preparation for coding tests  
reference: [https://covenant.tistory.com/224](https://covenant.tistory.com/224)

<br>

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