# FedL-Transformer_Wireless
Fedreated Transformer over Wireless Communication

Enable packet loss in OpenWRT
```shell
tc qdisc add dev wlan1 root netem loss 5%
tc qdisc show dev wlan1
```

Remove packet loss
```shell
tc qdisc del dev wlan1 root
```