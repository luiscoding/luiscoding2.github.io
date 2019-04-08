---
layout:     post
title:      Docker Image On Linux Server
subtitle:    
date:       2018-09-17
author:     Lu Zhang
header-img: img/images/img-linklist.jpg
catalog: true
tags:
    - Linux
---
## Why to use Docker Image?
At first, when I worked in the spatial computing lab, the demo was built and packed in a docker, which is convenient to install and run. Later, when I joined another group, which focus on the table understanding, we have a server there, but when I try to use the server to run my script, I found several problems: 
1. Permission to run apt-get 
2. Unstable enviroment
3. Regular Clean

So docker could be a choice at for this case. 
## What is Docker?

![img](https://cdn-images-1.medium.com/max/1600/1*easlVE_DOqRDUDkVINRI9g.png)

Docker can be seen as a isolated environment
Docker can be seen as a file which can be used to build the environment 

From the image above, we can see, 
First, we have a Dockerfile:
In the Dockerfile, we write the code,  which contain information like what app we want to install in our environment, the following one is a dockerfile which install the firefox-headless 

```Dockerfile 
# Run Firefox Headless in a container
#
#
# To run (without seccomp):
# docker run -d -p 6000:6000 --cap-add=SYS_ADMIN justinribeiro/firefox-headless
#

# Base docker image
FROM debian:sid
LABEL name="firefox-headless" \
			maintainer="Justin Ribeiro <justin@justinribeiro.com>" \
			version="1.0" \
			description="Firefox Headless in a container"

# Install deps + add Chrome Stable + purge all the things
RUN apt-get update && apt-get install -y \
	apt-transport-https \
	ca-certificates \
  gnupg \
	firefox \
	--no-install-recommends \
	&& apt-get purge --auto-remove -y curl gnupg \
	&& rm -rf /var/lib/apt/lists/*

RUN mkdir -p /etc/firefox/
RUN echo '\
    lockPref("devtools.debugger.force-local", false);\n\
    lockPref("devtools.debugger.remote-enabled", true);\n'\
		>> /etc/firefox/syspref.js

# Add firefox as a user
RUN groupadd -r firefox && useradd -r -g firefox -G audio,video firefox \
    && mkdir -p /home/firefox && chown -R firefox:firefox /home/firefox \
    && mkdir -p /home/firefox/.mozilla \
    && chown -R firefox:firefox /home/firefox/.mozilla

# Run firefox non-privileged
USER firefox

# Expose port 6000
EXPOSE 6000

# Autorun chrome headless with no GPU
ENTRYPOINT [ "firefox" ]
CMD [ "--start-debugger-server","--headless"]
```

-- To be continued 