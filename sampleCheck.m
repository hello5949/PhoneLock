clear all;
% A = csvread("C:\Users\EN301\Desktop\Rong\Phone_Lock\sample.csv",1,1);
% data_u = zeros(5,8192);
% for j = 1:5
%     for i = 1:8192
%         data_u(j,i) = mean(A( 240*(j-1)+1 : 240*j, i));
%     end
% end
% data_o = zeros(1200);
% for j = 1:1200
%     dd = 0;
%     for i = 1:8192
%         dd = dd + abs((A(j,i)-data_u(ceil(j/240),i)));
%     end
%     data_o(j) = dd/8192;
% end
% 
% lossList = csvread("C:\Users\EN301\Desktop\Rong\Phone_Lock\lossList.csv",0,0);
% train_o = [data_o(1:200,1); data_o(241:440,1); data_o(481:680,1); data_o(721:920,1); data_o(961:1160,1)];
%     subplot(2,1,1);
% for i = 1:5
%     plot(train_o);
%     grid on;
%     set(gca,'xtick',[1,200,400,600,800])
%     ylabel('MAE')
%     xlabel('iPhone-7P                iPhone-8P                 Samsung                    LG                    iPhone-XR');
% end
%     subplot(2,1,2);
% for i = 1:5
%     plot(lossList(i,:));
%     grid on;
%     set(gca,'xtick',[1,200,400,600,800])
%     ylabel('loss')
%     xlabel('iPhone-7P                iPhone-8P                 Samsung                    LG                    iPhone-XR');
%     hold on
% end
%     legend('iPhone-7P', 'iPhone-8P', 'Samsung', 'LG', 'iPhone-XR')

%%% TIME DOMAIN %%%
A = csvread("C:\Users\EN301\Desktop\Rong\Phone_Lock\Sample\sample_timeDomain.csv",0,0,[0 0 19999999 0]);
data = zeros(5,240,16384);
for i = 1:5
    for j = 1:240
        data(i,j,:) = A((i-1)*2000000+(j-1)*16384+1 : i*2000000+j*16384)
    end
end
% data_time_u = zeros(5,16384);
% for k = 1:5
%     for i = 1:240
%         for j = 1:16384
%             data_time_u(240*(k-1)+i,j) = A
%             plot(A(16384*(i-1)+1 : 16384*i,1));
%             hold on;
%         end
%     end
% end

