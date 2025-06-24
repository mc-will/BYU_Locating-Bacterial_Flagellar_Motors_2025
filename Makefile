cheer_up:
	@echo 'we gotta catch them all !!'


tree:
	tree -d -L 3

# TODO :
# commande pour charger les images / upload toutes les images au même endroit ?
# commande pour charger les modèles / upload tous les modèles au même endroit ?

# TODO déprécier
create_src_structure:
	mkdir -p src/utils
	mkdir -p src/ml_logic/train
	mkdir -p src/ml_logic/process


pylint:
  @find . -iname "*.py" -not -path "./tests/*" | xargs -n1 -I {} pylint --output-format=colorized {}; true

.PHONY: rename

rename_processed_pictures:
	@for dir in data/pictures_process/mean_image data/pictures_process/adaptequal_1 \
		data/pictures_process/adaptequal_01 data/pictures_process/adaptequal_05; do \
		if [ -d "$$dir" ]; then \
			echo "Renaming files in $$dir..."; \
			for file in "$$dir"/*.jpg; do \
				[ -e "$$file" ] || continue; \
				base=$$(basename "$$file"); \
				newname=$$(echo "$$base" | sed -E 's/.*_([^_]+_[^_]+\.jpg)/\1/'); \
				mv "$$file" "$$dir/$$newname"; \
			done; \
		else \
			echo "Directory $$dir does not exist. Skipping..."; \
		fi; \
	done
