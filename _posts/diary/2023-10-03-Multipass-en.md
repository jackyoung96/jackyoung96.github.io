---
layout: post
title: Diary - Using Ubuntu on Mac OS (Apple silicon M1)
tags: archive
lang: en
---

If I had to pick just one thing that became most inconvenient after leaving the lab, it would be that the Linux computer that only I could use disappeared. I have no personal GPU either... There are so many open-source projects that run on Ubuntu, so even when I want to try out various tutorials, there are many times it just won't work.
But I'd always thought I couldn't dual-boot my precious MacBook, yet it's so inconvenient I can't stand it. There's also the option of using a VM, but I found a slightly more convenient method. It's `Multipass`.

## What is Multipass?

> Multipass is a virtual machine solution provided by Canonical, the developer of Ubuntu. It's a virtual machine solution Canonical created in 2019. It supports a Terminal-based interface, but in exchange its advantage is very low resource consumption.

I was immediately drawn in by the phrase **very low resource consumption?** and decided to install it.

The installation method is simple.

```bash
brew install --cask multipass
```
Still, it takes quite a while because you have to install the linux images.
Once installation is complete, you can look up the list of images.
```bash
multipass find
```
<img width="771" alt="image" src="https://github.com/snulion-study/algorithm-adv/assets/57203764/cf64cc5b-6ea3-4caa-83b4-be570693e735">
The fun part is that it provides ROS. It makes me wonder if robotics has come this far. (Did I run away and abandon it for nothing?)

The image I wanted was 18.04, which doesn't show up here. But additional installation is possible, so don't worry.

```bash
# multipass launch [Image] -n [VM name] -c [CPU cores] -m [memory] -d [disk space]
multipass launch 18.04 -n linux -c 4 -m 4G -d 20G
```
Through the command above, install Ubuntu 18.04 under the virtual machine name "linux." For memory, disk space, and CPU count, just pick appropriately as you see fit. (The more you use, the more your MacBook's performance drops while it's running.)

Now you can access the linux shell and use Ubuntu 18.04.

```bash
# multipass shell [VM name]
multipass shell linux
```

<img width="431" alt="image" src="https://github.com/snulion-study/algorithm-adv/assets/57203764/092eb326-9479-4cf1-bb95-15a6ad5c2bcc">

Oh yeeahhh~~

## Using Multipass in vscode

Since I can't code without VScode, I'll look into how to work on the multipass environment through vscode.

First, create a public key for the ssh connection.
```
ssh-keygen -t rsa
```
Just press enter to skip through all the prompts that come up (key generation location, passphrase). Then you can confirm that a `~/.ssh/id_rsa.pub` file has been created. If you open it, you can confirm that the public key has been generated.
Now, using the public key, create a yaml file. The contents of the yaml file are as follows.

```yaml
groups:
  - vscode
runcmd:
  - adduser ubuntu vscode
ssh_authorized_keys:
  - ssh-rsa <public key>
```
Just copy the generated public key into the public key field. Use this yaml file to launch the VM again.
```bash
multipass launch 18.04 -n linux -c 4 -m 2G -d 20GB --cloud-init vscode.yaml
```
It's now set up to allow ssh access through vscode!
Then check the virtual machine's IP with the command below.
```bash
multipass info linux
```

Now go back to your local location and configure the ssh config.
Open the `~/.ssh/config` file and add the content below.

<img width="159" alt="image" src="https://github.com/snulion-study/algorithm-adv/assets/57203764/696fc1d0-a7a7-473c-ad8b-95620cb29ba2">

With this, you can access the ubuntu virtual machine through ssh!
Finally, mount the local storage.

```bash
multipass mount [desired local path] [VM name]:[VM path]
```

After a successful mount, a "permission denied" error may occur, which is due to the MacBook's permission issue regarding disk access.
You can resolve it by going to `Settings -> Privacy & Security -> Full Disk Access` and granting permission to multipassd.

Now you can use the ubuntu environment through vscode on your MacBook!

## Uninstall 

If you keep the multipass environment running, it eats up memory, so it's good to turn it off when not in use.
```bash
multipass stop linux # Stops the virtual machine
multipass delete linux # Removes the virtual machine from management
multipass purge # Completely deletes the virtual machine
```
If you purge, you can't recover it. The rest are recoverable.
```bash
multipass start linux # Recovery from stop
multipass recover linux # Recovery from delete
```

The difference between stop and delete is as follows.
- stop: deletes the primary instance
- delete: deletes all instances

## References

https://elsainmac.tistory.com/870  
https://discourse.ubuntu.com/t/using-multipass-with-vscode/34905  
https://github.com/canonical/multipass/issues/1389   
https://multipass.run/docs/delete-command  
https://multipass.run/docs/stop-command
