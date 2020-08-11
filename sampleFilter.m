clear all;
%%%define%%%
filePath = "C:\Users\EN301\Desktop\Rong\Phone_Lock\Sample";
fileName = "sample_timeDomain.csv";
phoneNum = 5;
dataLen = 4000000;
sampleLen = 16384;
sampleNumber = 325;
Threshold = 12;
step = sampleLen/2;


if floor(dataLen/step) < sampleNumber
    sampleNumber = floor(dataLen/step);
end

A = csvread(filePath+"\"+fileName,0,0,[0 0 dataLen*phoneNum-1 0]);
data = zeros(phoneNum*sampleNumber,sampleLen/2);
data_time = zeros(phoneNum*sampleNumber,sampleLen);
for i = 1:phoneNum
    cc = A((i-1)*dataLen+1:i*dataLen);
    j = 0;
    count = 0;
    while j < sampleNumber
        j = j+1;
        check = max(cc(1+count:16384+count)) - min(cc(1+count:16384+count));
        if check > Threshold
            disp("check > Threshold "+num2str(i)+"   "+num2str(count));
            j = j-1;
        else
            dd = cc(1+count:16384+count) - min(cc(1+count:16384+count));
            cc_fft = abs(fft(dd));
            cc_fft(1) = 0;
            data((i-1)*sampleNumber+j,:) = cc_fft(1:sampleLen/2);
            data_time((i-1)*sampleNumber+j,:) = dd;
        end
        count = count + step;
        if (count + 16384) >= dataLen
            disp("類別 "+num2str(i)+" 樣本不足 "+num2str(j));
            break
        end
    end
end
csvwrite("sample_clear.csv", data);
csvwrite("sample_clear_time.csv", data_time);