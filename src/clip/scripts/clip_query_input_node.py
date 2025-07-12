#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
from std_msgs.msg import String

class ClipQueryInputNode:
    def __init__(self):
        rospy.init_node("clip_query_input_node", anonymous=True)
        self.pub = rospy.Publisher("/adv_robocup/sam2clip/clip_query", String, queue_size=10)
        self.current_prompt = ""
        self.is_active = False
        self.timer = rospy.Timer(rospy.Duration(1.0), self.timer_callback)
        self.chat_sub = rospy.Subscriber("/adv_robocup/chat_intent", String, self.chat_callback)

        rospy.loginfo("Clip Query Input Node started. Publishing at 1Hz.")
        rospy.loginfo("Please enter prompts (input 'stop' to stop publishing, 'exit' to quit):")
        
    def timer_callback(self, event):
        if self.is_active and self.current_prompt:
            self.pub.publish(String(data=self.current_prompt))

    def chat_callback(self, msg):
        content = msg.data.strip()
        if content.lower() == "stop":
            self.is_active = False
            rospy.loginfo("Stopped publishing due to chat_intent='stop'")
        else:
            self.current_prompt = content
            self.is_active = True
            rospy.loginfo(f"Received intent and updated prompt: \"{content}\"")

    def run(self):
        while not rospy.is_shutdown():
            try:
                prompt = input("> ").strip()
                if prompt.lower() == "exit":
                    rospy.loginfo("Exiting clip_query_input_node")
                    break
                elif prompt.lower() == "stop":
                    self.is_active = False
                    rospy.loginfo("Stopped publishing due to user input")
                elif prompt:
                    self.current_prompt = prompt
                    self.is_active = True
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
