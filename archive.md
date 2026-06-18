---
layout: page
title: Archive
lang_toggle: true
---

<div class="lang-section" data-lang="ko" markdown="1">
이 곳은 **개인적으로** 논문을 리뷰하고 관련 문헌들을 정리하는 공간입니다.
</div>
<div class="lang-section" data-lang="en" hidden markdown="1">
This is a **personal space** where I review papers and organize related documents.
</div>



<div id="wrap">
      <!-- Main content -->
    <div class="main-layout">
        <div id="container-search">
            <main>
              {% include search.html %}
            </main>
        </div>
    </div>
</div>

<div class="lang-section" data-lang="ko" markdown="1">

## 전체 목록

<br>

{% for post in site.posts %}
  {% if post.tags contains "archive" and post.lang != "en" %}
  {{ post.date | date_to_string }} &raquo; [ {{ post.title }} ]({{ post.url }})
  {% endif %}
{% endfor %}

</div>

<div class="lang-section" data-lang="en" hidden markdown="1">

## All posts

<br>

{% assign en_posts = site.posts | where: "lang", "en" %}
{% for post in en_posts %}
  {% if post.tags contains "archive" %}
  {{ post.date | date_to_string }} &raquo; [ {{ post.title }} ]({{ post.url }})
  {% endif %}
{% else %}
  *English translations coming soon.*
{% endfor %}

</div>