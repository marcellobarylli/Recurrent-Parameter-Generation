mkdir checkpoint_test
mkdir checkpoint_train
mkdir generated

mv ./checkpoint/*class0314* ./checkpoint_test
mv ./checkpoint/*class0482* ./checkpoint_test
mv ./checkpoint/*class0589* ./checkpoint_test
mv ./checkpoint/*class0197* ./checkpoint_test
mv ./checkpoint/*class0462* ./checkpoint_test
mv ./checkpoint/*class0111* ./checkpoint_test
mv ./checkpoint/*class0101* ./checkpoint_test
mv ./checkpoint/*class0278* ./checkpoint_test
mv ./checkpoint/*class0793* ./checkpoint_test
mv ./checkpoint/*class0279* ./checkpoint_test
mv ./checkpoint/*class0653* ./checkpoint_test
mv ./checkpoint/*class0238* ./checkpoint_test
mv ./checkpoint/*class1001* ./checkpoint_test
mv ./checkpoint/*class0141* ./checkpoint_test
mv ./checkpoint/*class0884* ./checkpoint_test
mv ./checkpoint/*class0592* ./checkpoint_test
mv ./checkpoint/*class0502* ./checkpoint_test
mv ./checkpoint/*class0643* ./checkpoint_test
mv ./checkpoint/*class0383* ./checkpoint_test
mv ./checkpoint/*class0128* ./checkpoint_test

mv ./checkpoint/* ./checkpoint_train

rm checkpoint -r