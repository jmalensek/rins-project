# RINS_PROJECT

Komande za zagon nav2 za fizičnega robota:

1. ros2 launch turtlebot4_navigation localization.launch.py map:=/home/kappa/Documents/Task1/src/rins_project/maps/map_name.yaml
2. ros2 launch turtlebot4_navigation nav2.launch.py params_file:=/home/kappa/Documents/Task1/src/rins_project/config/nav2_careless_extreme.yaml # Po želji zamenjaj z drugim nav2 configom
3. ros2 launch turtlebot4_viz view_navigation.launch.py
