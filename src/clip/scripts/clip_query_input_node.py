#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
from std_msgs.msg import String

def main():
    rospy.init_node("clip_query_input_node", anonymous=True)
    pub = rospy.Publisher("/adv_robocup/sam2clip/clip_query", String, queue_size=10)
    rospy.loginfo("Please enter prompts (input 'exit' to quit):")

    while not rospy.is_shutdown():
        try:
            prompt = input("> ").strip()
            if prompt.lower() == "exit":
                rospy.loginfo("Exiting clip_query_input_node")
                break
            if prompt:
                pub.publish(String(data=prompt))
                rospy.loginfo(f"Published prompt: \"{prompt}\"")
        except (EOFError, KeyboardInterrupt):
            rospy.loginfo("Exiting clip_query_input_node")
            break

if __name__ == "__main__":
    main()
