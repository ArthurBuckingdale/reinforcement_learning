function [rotation_matrix]=calculate_rot_matrix(x)
%the purpose of this function is calculating the rotation
%matrix for each step. We will calculate the force in the frame
%of the body, then transform it to the inertial frame.
th_ = x(7);
ph_ = x(8);
rh_ = x(9);
rotation_matrix(1,1)=(cos(th_)*cos(ph_))-(cos(th_)*sin(ph_)*sin(rh_));
rotation_matrix(1,2)=(-cos(rh_)*sin(ph_))-(cos(ph_)*cos(th_)*sin(rh_));
rotation_matrix(1,3)=sin(th_)*sin(rh_);
rotation_matrix(2,1)=(cos(th_)*cos(rh_)*sin(ph_))+(cos(ph_)*sin(rh_));
rotation_matrix(2,2)=(cos(ph_)*cos(th_)*cos(rh_))-(sin(ph_)*sin(rh_));
rotation_matrix(2,3)=(-cos(rh_)*sin(th_));
rotation_matrix(3,1)=sin(ph_)*sin(th_);
rotation_matrix(3,2)=cos(ph_)*sin(th_);
rotation_matrix(3,3)=cos(th_);
end