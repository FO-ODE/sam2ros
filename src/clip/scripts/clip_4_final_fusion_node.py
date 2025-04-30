#!/usr/bin/env python
# -*- coding: utf-8 -*-
import rospy
from std_msgs.msg import String

class FinalFusionNode:
    def __init__(self):
        rospy.init_node('final_fusion_node', anonymous=True)

        self.motion_state = None
        self.action_group = None
        self.fine_action = None

        rospy.Subscriber("/motion_state", String, self.motion_callback)
        rospy.Subscriber("/action_group", String, self.group_callback)
        rospy.Subscriber("/fine_behavior_result", String, self.fine_callback)

        self.pub = rospy.Publisher("/final_behavior_result", String, queue_size=10)

        rospy.Timer(rospy.Duration(1.0), self.publish_fused_result)  # 每秒发布一次融合结果

        rospy.loginfo("[Fusion] Final Fusion Node started.")
        rospy.spin()

    def motion_callback(self, msg):
        self.motion_state = msg.data

    def group_callback(self, msg):
        self.action_group = msg.data

    def fine_callback(self, msg):
        self.fine_action = msg.data

    def publish_fused_result(self, event):
        if self.motion_state and self.action_group and self.fine_action:
            result = f"[Behavior] {self.motion_state} → {self.action_group} → {self.fine_action}"
            self.pub.publish(result)
            rospy.loginfo(result)

if __name__ == '__main__':
    try:
        FinalFusionNode()
    except rospy.ROSInterruptException:
        pass
