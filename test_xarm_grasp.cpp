#include <signal.h>
#include <chrono>
#include <rclcpp/rclcpp.hpp>
#include <xarm_api/xarm_ros_client.h>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <std_msgs/msg/string.hpp>
#include <memory>
#include <cmath>
#include <queue>
#include <vector>
#include <mutex>
#include <thread>

using namespace std::chrono_literals;

void exit_sig_handler(int signum) {
    fprintf(stderr, "[xarm_driver] Ctrl-C caught, exit process...\n");
    exit(-1);
}

class XArmControl : public rclcpp::Node {
public:
    XArmControl(std::shared_ptr<xarm_api::XArmROSClient> client) 
        : Node("xarm_control"), client_(client) {
        // 初始化默认位置参数
        initial_pose_ = {259.7, -50, 412.4, 180, 0, -45.2};
        box_pose_ = {365.3, -240.2, 294.6, 180, 0, -45.2};
        initial_joint_angles_ = {0, 0, 0, 0, 0.789, 0, 0};
        
        gripper_pub_ = this->create_publisher<std_msgs::msg::String>("gripper_commands", 10);
        
        target_sub_ = this->create_subscription<geometry_msgs::msg::PoseStamped>(
            "/robot_target_pose", 10,
            [this](const geometry_msgs::msg::PoseStamped::SharedPtr msg) {
                handle_new_target(msg);
            });
            
        container_sub_ = this->create_subscription<geometry_msgs::msg::PoseStamped>(
            "/container_pose", 10,
            [this](const geometry_msgs::msg::PoseStamped::SharedPtr msg) {
                std::lock_guard<std::mutex> lock(bbox_pose_mutex_);
                box_pose_[0] = msg->pose.position.x * 1000;
                box_pose_[1] = msg->pose.position.y * 1000;
                RCLCPP_INFO(this->get_logger(), "Container position updated to [%.1f, %.1f]", 
                          box_pose_[0], box_pose_[1]);
            });
    }

    void reset_state() {
        std::lock_guard<std::mutex> lock(target_queue_mutex_);
        is_first_target_ = true;
        is_executing_ = false;
        std::queue<std::vector<float>> empty;
        std::swap(target_queue_, empty);
        RCLCPP_INFO(this->get_logger(), "System state has been reset");
    }

private:
    std::shared_ptr<xarm_api::XArmROSClient> client_;
    rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr target_sub_;
    rclcpp::Publisher<std_msgs::msg::String>::SharedPtr gripper_pub_;
    rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr container_sub_;
    std::mutex bbox_pose_mutex_;
    
    std::vector<float> initial_pose_;
    std::vector<float> box_pose_;
    std::vector<float> initial_joint_angles_;
    
    std::queue<std::vector<float>> target_queue_;
    std::mutex target_queue_mutex_;
    bool is_executing_ = false;
    bool is_first_target_ = true;
    bool should_return_home_ = false;

    float normalize_angle(float angle) {
        while(angle > M_PI) angle -= 2*M_PI;
        while(angle < -M_PI) angle += 2*M_PI;
        return angle;
    }

    void handle_new_target(const geometry_msgs::msg::PoseStamped::SharedPtr msg) {
        std::vector<float> target_pose = {
            static_cast<float>(msg->pose.position.x * 1000),
            static_cast<float>(msg->pose.position.y * 1000),
            static_cast<float>(200),
            -180.0f, 0.0f,
            static_cast<float>(msg->pose.orientation.z)
        };
        
        {
            std::lock_guard<std::mutex> lock(target_queue_mutex_);
            target_queue_.push(target_pose);
        }
        
        if (!is_executing_) {
            process_target_queue();
        }
    }

    void process_target_queue() {
        is_executing_ = true;
        
        std::thread([this]() {
            is_first_target_ = true;
            should_return_home_ = false;
            
            while (true) {
                std::vector<float> target_pose;
                bool has_more_targets = false;
                
                {
                    std::lock_guard<std::mutex> lock(target_queue_mutex_);
                    if (target_queue_.empty()) {
                        is_executing_ = false;
                        if (should_return_home_) {
                            return_to_initial();
                        }
                        return;
                    }
                    
                    target_pose = target_queue_.front();
                    target_queue_.pop();
                    has_more_targets = !target_queue_.empty();
                    should_return_home_ = !has_more_targets;
                }
                
                execute_target_sequence(target_pose);
            }
        }).detach();
    }

    std::vector<float> calculate_smart_prepare_pose(const std::vector<float>& target_pose) {
        std::lock_guard<std::mutex> lock(bbox_pose_mutex_);
        float mid_x = (target_pose[0] * 0.5 + box_pose_[0] * 0.5);
        float mid_y = (target_pose[1] * 0.5 + box_pose_[1] * 0.5);
        float safe_height = box_pose_[2];
        
        return {mid_x, mid_y, safe_height, box_pose_[3], box_pose_[4], 0.0f};
    }

    void execute_target_sequence(const std::vector<float>& target_pose) {
        // 更新盒子位置的Z旋转角度
        {
            std::lock_guard<std::mutex> lock(bbox_pose_mutex_);
            box_pose_[5] = target_pose[5];
        }
        
        if (is_first_target_) {
            RCLCPP_INFO(this->get_logger(), "First target, moving to initial position");
            if (!move_to_pose(initial_pose_, false)) return;
            is_first_target_ = false;
        }
        
        RCLCPP_INFO(this->get_logger(), "Moving to target position");
        if (!move_to_target(target_pose)) return;
        
        publish_gripper_command("g");
        rclcpp::sleep_for(4s);

        {
            std::lock_guard<std::mutex> lock(bbox_pose_mutex_);
            RCLCPP_INFO(this->get_logger(), "Moving to box position");
            if (!move_to_pose(box_pose_, true)) return;
        }

        publish_gripper_command("r");
        rclcpp::sleep_for(4s);

        if (!should_return_home_) {
            auto prepare_pose = calculate_smart_prepare_pose(target_pose);
            RCLCPP_INFO(this->get_logger(), "Moving to smart prepare position");
            if (!move_to_pose(prepare_pose, false)) return;
        }
    }

    void return_to_initial() {
        RCLCPP_INFO(this->get_logger(), "Returning to initial position");
        if (!move_to_pose(initial_pose_, false)) {
            RCLCPP_ERROR(this->get_logger(), "Failed to return to initial pose!");
        }
    }

    void publish_gripper_command(const std::string& command) {
        auto msg = std_msgs::msg::String();
        msg.data = command;
        gripper_pub_->publish(msg);
    }

    bool move_to_target(const std::vector<float>& target_pose) {
        std::vector<float> current_pose;
        if(client_->get_position(current_pose) != 0) return false;

        // XY移动
        std::vector<float> xy_target = current_pose;
        xy_target[0] = target_pose[0];
        xy_target[1] = target_pose[1];
        if(client_->set_position(xy_target, 0, 180, 1000, 0, true, 10.0f) != 0) return false;

        // Z旋转
        std::vector<float> current_joints;
        if(client_->get_servo_angle(current_joints) != 0) return false;
        
        std::vector<float> joint_target = current_joints;
        joint_target[4] = normalize_angle(joint_target[4] + target_pose[5]);
        if(client_->set_servo_angle(joint_target, 1, 1000, 0, true, 10.0f) != 0) return false;

        // Z移动
        if(client_->get_position(current_pose) != 0) return false;
        std::vector<float> z_target = current_pose;
        z_target[2] = target_pose[2];
        return (client_->set_position(z_target, 0, 180, 1000, 0, true, 10.0f) == 0);
    }

    bool move_to_pose(const std::vector<float>& target_pose, bool apply_z_rotation) {
        std::vector<float> current_pose;
        if(client_->get_position(current_pose) != 0) return false;

        // Z移动
        std::vector<float> z_target = current_pose;
        z_target[2] = target_pose[2];
        if(client_->set_position(z_target, 0, 180, 1000, 0, true, 10.0f) != 0) return false;

        // XY移动
        std::vector<float> xy_target = z_target;
        xy_target[0] = target_pose[0];
        xy_target[1] = target_pose[1];
        if(client_->set_position(xy_target, 0, 180, 1000, 0, true, 10.0f) != 0) return false;

        // Z旋转（仅在需要时执行）
        if(apply_z_rotation) {
            std::vector<float> current_joints;
            if(client_->get_servo_angle(current_joints) != 0) return false;
            
            std::vector<float> joint_target = current_joints;
            joint_target[4] = normalize_angle(joint_target[4] - target_pose[5]);
            return (client_->set_servo_angle(joint_target, 1, 1000, 0, true, 10.0f) == 0);
        }
        
        return true;
    }
};

int main(int argc, char **argv) {
    rclcpp::init(argc, argv);
    signal(SIGINT, exit_sig_handler);
    
    auto node = rclcpp::Node::make_shared("xarm_ros_client");
    std::string hw_ns = "xarm";
    bool sim_mode = false;
    
    auto client = std::make_shared<xarm_api::XArmROSClient>();
    client->init(node, hw_ns);
    client->clean_error();
    client->set_simulation_robot(sim_mode);
    client->motion_enable(!sim_mode);
    client->set_mode(0);
    client->set_state(0);
    
    rclcpp::sleep_for(1s);
    auto xarm_control = std::make_shared<XArmControl>(client);
    rclcpp::spin(xarm_control);
    rclcpp::shutdown();
    return 0;
}