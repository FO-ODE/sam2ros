#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
from std_msgs.msg import String

class ClipQueryInputNode:
    def __init__(self):
        rospy.init_node("clip_query_input_node", anonymous=True)
        self.pub = rospy.Publisher("/adv_robocup/sam2clip/clip_query", String, queue_size=10)
        self.current_prompt = ""
        self.timer = rospy.Timer(rospy.Duration(1.0), self.timer_callback)  # 1Hz
        rospy.loginfo("Clip Query Input Node started. Publishing at 1Hz.")
        rospy.loginfo("Please enter prompts (input 'exit' to quit):")
        
    def timer_callback(self, event):
        if self.current_prompt:
            self.pub.publish(String(data=self.current_prompt))
            # rospy.loginfo(f"Published prompt: \"{self.current_prompt}\"")
    
    def run(self):
        while not rospy.is_shutdown():
            try:
                prompt = input("> ").strip()
                if prompt.lower() == "exit":
                    rospy.loginfo("Exiting clip_query_input_node")
                    break
                if prompt:
                    self.current_prompt = prompt
                    rospy.loginfo(f"Publishing prompt with 1Hz: \"{prompt}\"")
            except (EOFError, KeyboardInterrupt):
                rospy.loginfo("Exiting clip_query_input_node")
                break

def main():
    try:
        node = ClipQueryInputNode()
        node.run()
    except rospy.ROSInterruptException:
        pass

if __name__ == "__main__":
    main()
