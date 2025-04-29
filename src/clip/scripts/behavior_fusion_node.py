#!/usr/bin/env python
# -*- coding: utf-8 -*-
import rospy
from std_msgs.msg import String

class BehaviorFusionNode:
    def __init__(self):
        rospy.init_node('behavior_fusion_node', anonymous=True)

        self.stage1_action = None
        self.stage2_action = None

        # 订阅阶段1、阶段2的推理结果
        rospy.Subscriber("/clip_stage1_result", String, self.stage1_callback, queue_size=10)
        rospy.Subscriber("/clip_stage2_result", String, self.stage2_callback, queue_size=10)

        # 发布最终融合的行为结果
        self.pub = rospy.Publisher("/final_behavior_result", String, queue_size=10)

        rospy.loginfo("Behavior Fusion Node Started.")
        rospy.spin()

    def stage1_callback(self, msg):
        self.stage1_action = msg.data
        self.publish_fusion_result()

    def stage2_callback(self, msg):
        self.stage2_action = msg.data
        self.publish_fusion_result()

    def publish_fusion_result(self):
        # 当两个阶段结果都到达时，融合发布
        if self.stage1_action and self.stage2_action:
            final_behavior = f"{self.stage1_action} - {self.stage2_action}"
            self.pub.publish(final_behavior)
            rospy.loginfo(f"[Fusion] Final Behavior: {final_behavior}")

if __name__ == '__main__':
    try:
        BehaviorFusionNode()
    except rospy.ROSInterruptException:
        pass