function rot_mat_unit_test()
%the purpose of this function is to validate a coordinate system for its
%rotations. The coordinate system is defined here and this will check that
%the rotation matrix is correctly defined. Here we go. This is geared for
%the three dimensional rocket lander so we'll be taking the 7,8,9th
%elements of a vector to calculate the transformation
close all
clear all

%% defining the coordinate system
%so we will define the coordinate system here
%+Z is upwards on screen -Z is downwards on the screen 
%+Y is outwards of the screen, -Y is into the screen
%+X is to the right of the screen, -X is to the left of the screen
%rotation about the X axis (pitch) is positive when counter clockwise
%   negative when clockwise. Labelled theta
%rotation about the Y axis (roll) is positive when counter clockwise
%   negative when clockwise. Labelled phi
%rotation about the Z axis (yaw) is positive when counter clockwise
%   negative when clockwise. Labelled rho

%% unit test 1 
%first, we a roll. This implies a force positive in Z upon a 90 degree
%rotation about the y axis will give us a -X force
test_one=[0,0,1];
storage_vector1=[0,0,0,0,0,0,0,pi/2,0];
[rotation_matrix]=calculate_rot_matrix(storage_vector1);

unit_result=test_one*rotation_matrix;
unit_result2=rotation_matrix*test_one';
disp(rotation_matrix)
disp('-----------------------')
disp(unit_result)
if unit_result(1) == -1
    disp('Unit test for rolling passed')
else
    disp('Unit test for rolling failed. Go fuck yourself you big idiot')
end

%% unit test 2 
%second, we test a pitch. this implies a positive force in Z upon a 90
%degree pitch, will become a positive force in Y 
test_two=[0,0,1];
storage_vector=[0,0,0,0,0,0,pi/2,0,0];
[rotation_matrix2]=calculate_rot_matrix(storage_vector);

unit_result_two=test_two*rotation_matrix2;
unit_result2_two=rotation_matrix2*test_two';
disp(rotation_matrix2)
disp('-----------------------')
disp(unit_result_two)
if unit_result_two(2) == 1
    disp('Unit test for pitch passed')
else
    disp('Unit test for pitch failed. Go fuck yourself you big idiot')
end

%% unit test 3 
%third we test our yaw-ing a +X force will become a +Y force under a -90
%degree rotation
test_three=[1,0,0];
storage_vector3=[0,0,0,0,0,0,0,0,-pi/2];
[rotation_matrix3]=calculate_rot_matrix(storage_vector3);

unit_result_three=test_three*rotation_matrix3;
unit_result2_three=rotation_matrix3*test_three';
disp(rotation_matrix3)
disp('-----------------------')
disp(unit_result_three)
if unit_result_three(2) == 1
    disp('Unit test for yaw passed')
else
    disp('Unit test for yaw failed. Go fuck yourself you big idiot')
end
