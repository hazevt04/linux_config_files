# .bashrc for CentOS thin client box

# Source global definitions
if [ -f /etc/bashrc ]; then
	. /etc/bashrc
fi

# Uncomment the following line if you don't like systemctl's auto-paging feature:
# export SYSTEMD_PAGER=

# User specific aliases and functions
if [[ -f ~/.bashrc.$USER ]]; then
	. ~/.bashrc.$USER
fi

