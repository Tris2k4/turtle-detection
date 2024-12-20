import argparse

import util


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--class-list', default='./class.names')
	parser.add_argument('--data-dir', default='./yolo-dataset')
	parser.add_argument('--output-dir', default='./output')
	parser.add_argument('--device', default='gpu')
	parser.add_argument('--learning-rate', default=0.00025)
	parser.add_argument('--batch-size', default=4)
	parser.add_argument('--iterations', default=20000)
	parser.add_argument('--checkpoint-period', default=500)
	parser.add_argument('--model', default='COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml')

	args = parser.parse_args()

	util.train(args.output_dir,
			   args.data_dir,
			   args.class_list,
			   device=args.device,
			   learning_rate=float(args.learning_rate),
			   batch_size=int(args.batch_size),
			   iterations=int(args.iterations),
			   checkpoint_period=int(args.checkpoint_period),
			   model=args.model)