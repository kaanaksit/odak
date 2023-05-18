# Fundamentals and Standards


This chapter will review some background information you will likely use in the rest of this course.
In addition, we will also introduce you to a structure where we establish some standards to decrease the chances of producing buggy or incompatible codes.


## Required Production Environment :material-alert-decagram:{ .mdx-pulse title="Too important!" }


:octicons-info-24: Informative ·
:octicons-beaker-24: Practical


We have provided some information in [prerequisites](index.md).
This information includes programming language requirements, required libraries, text editors, build environments, and operating system requirements.
For installing our library, odak, we strongly advise using the version in the source repository.
You can install odak from the source repository using your favorite terminal and operating system:


```shell
pip3 install git+https://github.com/kaanaksit/odak
```

Note that your production environment meaning your computer and required software for this course is important.
To avoid wasting time in the next chapters and get the most from this lecture, please ensure that you have dedicated enough time to set everything up as it should.


## Production Standards :material-alert-decagram:{ .mdx-pulse title="Too important!" }


:octicons-info-24: Informative


In this course, you will be asked to code and implement phenomena related to the physics of light.
Your work, meaning your production, should strictly follow certain habits to help build better tools and developments.


### Subversion and Revision Control


:octicons-info-24: Informative ·
:octicons-beaker-24: Practical

As you develop your code for your future homework and projects, you will discover that many things could go wrong.
For example, the hard drive that contains the only copy of your code could be damaged, or your most trusted friend (so-called) can claim that she compiled most of the work, although that is not the case.
These are just a few cases that may happen in your case.
Poor code control can cause companies to lose money by releasing incorrect codes or researchers to lose their reputations as their work is challenging to replicate.
_How do you claim in that case that you did your part?_
_What is the proper method to avoid losing data, time, effort, and motivation?_


This is where the subversion, authoring, and revision control systems come in for the example cases discussed in the previous paragraph.
In today's world, [Git](https://git-scm.com/) is a widespread version control system adopted by major websites such as [GitHub](https://github.com/) or [Gitlab](https://about.gitlab.com/).
We will not dive deep into how to use Git and all its features, but I will try to highlight parts that are likely essential for your workflow.
I encourage you to **use Git** for creating a repository for every one of your tasks.
You can either keep this repository in your locally and constantly back up somewhere else or use these online services such as [GitHub](https://github.com/) or [Gitlab](https://about.gitlab.com/).
I also encourage you to use the online services if you are a beginner.


Let us imagine that you want to start a repository on GitHub.
Make sure to create a private repository, and please only go public with any repository once you feel it is at a state where it can be shared with others.
Once you have created your repository on GitHub, you can clone the repository using the following command in a terminal:


```shell
git clone REPLACEWITHLOCATIONOFREPO
```


You can find out about the repository's location by visiting the repository's website that you have created.
The location is typically revealed by clicking the code button, as depicted in the below screenshot.


<figure markdown>
  ![Image title](media/git_clone.png){ width="600" }
  <figcaption>A screenshot showing how you can acquire the link for cloning a repository from GitHub.</figcaption>
</figure>


For example, in the above example case, the command should be updated with the following:

```shell
git clone git@github.com:kaanaksit/odak.git
```

If you want to share your private repository with someone you can go into the settings of your repository in its webpage and navigate to the collaborators section.
This way, you can assign roles to your collaborators that best suits your scenario.

!!! warning end "Secure your account"
    If you are using GitHub for your development, I highly encourage you to consider using [two-factor authentication](https://docs.github.com/en/authentication/securing-your-account-with-two-factor-authentication-2fa).


## Background Refresher :material-alert-decagram:{ .mdx-pulse title="Too important!" }


:octicons-info-24: Informative
