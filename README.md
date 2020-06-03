#[Deeplab v3+ cho bài toán phân đoạn não người]()

Đã sử dụng Cross entropy loss cho IoU cho class bằng 0.10

Sử dụng dice loss thì kết quả cao hơn chút là 0.13
 
**Dice loss**
 
 Dice loss là một hàm loss sử dụng phổ biến cho bài toán image segmentation dựa trên Dice coefficient(sử dụng đế đánh giá độ trùng lặp giữa 2 mẫu)
 os
 Khoảng giá trị cho Dice loss nằm giữa [0, 1] với 1 là giá trị mà overlap hoàn hảo nhất. 
  
 <img src="https://render.githubusercontent.com/render/math?math=e^{i \pi} = -1">