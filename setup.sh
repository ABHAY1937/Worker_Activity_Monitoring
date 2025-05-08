#!/bin/bash

#Create script.sh
cat <<EOF > /home/abhay/Music/Worker_Activity_Monitoring/script.sh
#!/bin/bash
source /home/abhay/anaconda3/bin/activate v9yolo  
python /home/abhay/Music/Worker_Activity_Monitoring/worker_activity.py
EOF

chmod +x /home/abhay/Music/Worker_Activity_Monitoring/script.sh

#Create activity.service
cat <<EOF > /home/abhay/Music/Worker_Activity_Monitoring/activity.service
[Unit]
Description=Run Python Script in Conda Env
After=network.target

[Service]
ExecStart=/home/abhay/Music/Worker_Activity_Monitoring/script.sh
WorkingDirectory=/home/abhay/Music/Worker_Activity_Monitoring
StandardOutput=inherit
StandardError=inherit
Restart=always
User=abhay

[Install]
WantedBy=multi-user.target
EOF

#Copy service to systemd
sudo cp /home/abhay/Music/Worker_Activity_Monitoring/activity.service /etc/systemd/system/

#Reload systemd and enable the service
sudo systemctl daemon-reload
sudo systemctl enable activity.service
sudo systemctl start activity.service

#Show service status
sudo systemctl status activity.service