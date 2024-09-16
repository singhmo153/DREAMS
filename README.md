# DREAMS(Diverse Reactions of Engagement and Attention Mind States Dataset)


Generate the OpenFace and MARLIN feature files in the following way:

![feature_extraction_and_statistical_feature_aggregation](https://github.com/user-attachments/assets/24129e0f-4369-4b81-b859-273323fa48be)



Note: 
1) For MARLIN, if the number of segments is less than 10 then add zero padding to make the total number of segments equal to 10. If number of segments is greater than 10 then trim the excess segments to retain only the first 10.
2) Stack all the videos' features into one npy file.
