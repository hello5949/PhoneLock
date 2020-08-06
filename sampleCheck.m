clear all;
A = csvread("C:\Users\EN301\Desktop\Rong\Phone_Lock\sample.csv",1,1);
data_u = zeros(5,8192);
for j = 1:5
    for i = 1:8192
        data_u(j,i) = mean(A( 240*(j-1)+1 : 240*j, i));
    end
end
data_o = zeros(1200);
for j = 1:1200
    dd = 0;
    for i = 1:8192
        dd = dd + abs((A(j,i)-data_u(ceil(j/240),i)));
    end
    data_o(j) = dd/8192;
end

lossList = csvread("C:\Users\EN301\Desktop\Rong\Phone_Lock\lossList.csv",0,0);
train_o = [data_o(1:200,1); data_o(241:440,1); data_o(481:680,1); data_o(721:920,1); data_o(961:1160,1)];
    subplot(4,1,1);
for i = 1:5
    plot(train_o);
    grid on;
    set(gca,'xtick',[1,200,400,600,800])
    title('Frequency Domain')
    ylabel('MAE')
    xlabel('iPhone-7P                iPhone-8P                 Samsung                    LG                    iPhone-XR');
end
    subplot(4,1,4);
for i = 1:5
    plot(lossList(i,:));
    grid on;
    set(gca,'xtick',[1,200,400,600,800])
    title('Train set Loss')
    ylabel('loss')
    xlabel('iPhone-7P                iPhone-8P                 Samsung                    LG                    iPhone-XR');
    hold on
end
    legend('iPhone-7P', 'iPhone-8P', 'Samsung', 'LG', 'iPhone-XR')

%%% TIME DOMAIN %%%
A = csvread("C:\Users\EN301\Desktop\Rong\Phone_Lock\Sample\sample_timeDomain.csv",0,0,[0 0 19999999 0]);
data = zeros(5,240,16384);
for i = 1:5
    cc = A((i-1)*4000000+1:i*4000000);
    for j = 1:240
        data(i,j,:) = cc((j-1)*16384+1:j*16384);
    end
end

data_time_u = zeros(5,16384);
u = zeros(16384);
for i = 1:5
    for j = 1:240
        for k = 1:16384
            u(k) = u(k) + data(i,j,k);
        end
    end
    for k = 1:16384
        u(k) = u(k)/240;
    end
    data_time_u(i,:) = u(k);
end

data_time_o = zeros(1200);
a1 = data(1,1,:);
a = reshape(data(1,1,:),[1,16384]);
b = data_time_u(1,:);
c = a-b;
for i = 1:5
    for j = 1:240
        mae = mean(abs(reshape(data(i,j,:),[1,16384])-data_time_u(i,:)));
        data_time_o((i-1)*240+j) = mae;
    end
end
train_data_time_o = [data_time_o(1:200,1); data_time_o(241:440,1); data_time_o(481:680,1); data_time_o(721:920,1); data_time_o(961:1160,1)];
subplot(4,1,2);

plot(train_data_time_o);
grid on;
set(gca,'xtick',[1,200,400,600,800])
title('Time Domain');
xlabel('Sample');
ylabel('MAE');

%%% MIN MAX%%%
min_max = zeros(1200);
for i = 1:5
    for j = 1:240
        min_max((i-1)*240+j) = max(data(i,j,:)) - min(data(i,j,:));
    end
end
train_minmax_time_o = [min_max(1:200,1); min_max(241:440,1); min_max(481:680,1); min_max(721:920,1); min_max(961:1160,1)];
subplot(4,1,3);
plot(train_minmax_time_o)
title('Max-Min');
xlabel('Count');
ylabel('Max-Min');

%%%plot histogram dirgram%%%
% histogram(min_max(:,1));
% title('Max-Min Histogram');
% xlabel('Count');
% ylabel('amount');

