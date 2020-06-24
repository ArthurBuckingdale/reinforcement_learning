function omega_matrix=obtain_angular_conversion_matrix(x)
%the purpose of this function is to correctly change our angles over from
%components into a vector. I'm not sure that I 100% understand this at the
%moment, but i'll get there. 

ph=x(7);
th=x(8);
rh=x(9);

omega_matrix(1,1)=1;
omega_matrix(1,2)=0;
omega_matrix(1,3)=-sin(th);
omega_matrix(2,1)=0;
omega_matrix(2,2)=cos(ph);
omega_matrix(2,3)=cos(th).*sin(ph);
omega_matrix(3,1)=0;
omega_matrix(3,2)=-sin(ph);
omega_matrix(3,3)=cos(th)*cos(ph);