if [ -z "$1" ]; then
  echo "Please provide a description."
  exit 1
fi

description="\"$1\""
output_dir="./output"
confirmed_output_dir="./confirmed_output"
file=$(find "$output_dir" -name 'results*')

if [ -n "$file" ]; then
  mv "$file" "$confirmed_output_dir/"
  filename=$(basename "$file")
#  quoted_filename="\"$filename\""
  echo "$description,$filename" >> ./confirmed_output/guide.csv
  echo "File moved and guide.csv updated."
else
  echo "No file matching results* found in $output_dir."
fi
