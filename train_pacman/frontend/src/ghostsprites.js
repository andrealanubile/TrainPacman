import * as constants from './constants'
import Spritesheet from './spritesheet';
import { Sprite } from 'pixi.js'

class GhostSprites extends Spritesheet {
    constructor(name, mode, direction) {
        super();
        this.name = name;
        this.mode = mode;
        this.direction = direction;
        this.ghostMap = {BLINKY: 0, PINKY: 2, INKY: 4, CLYDE: 6};
        this.sprite = new Sprite();
        this.sprite.anchor.set(0.5);
    }

    drawGhost(x, y, dt) {
        let gloc = this.ghostMap[this.name];
        let image;
        if (this.mode in [constants.SCATTER, constants.CHASE]) {
            switch (this.direction) {
                case constants.LEFT:
                    image = this.getImage(gloc, 8);
                    break;
                case constants.RIGHT:
                    image = this.getImage(gloc, 10);
                    break;
                case constants.DOWN:
                    image = this.getImage(gloc, 6);
                    break;
                case constants.UP:
                    image = this.getImage(gloc, 4);
                    break;
            }
        } else if (this.mode == constants.FRIGHT) {
            image = this.getImage(10, 4);
        } else if (this.mode == constants.SPAWN) {
            switch (this.direction) {
                case constants.LEFT:
                    image = this.getImage(8, 8)
                case constants.RIGHT:
                    image = this.getImage(8, 10);
                    break;
                case constants.DOWN:
                    image = this.getImage(8, 6);
                    break;
                case constants.UP:
                    image = this.getImage(8, 4);
                    break;
            }
        }
        this.sprite.x = (x + 1/2) * constants.TILEWIDTH;
        this.sprite.y = (y + 1/2) * constants.TILEHEIGHT;;
        this.sprite.texture = image;
    }

    getSprite() {
        return this.sprite;
    }

    getStartImage() {
        return this.getImage(this.ghostMap[this.name], 4)
    }


    getImage(x, y) {
        return super.getImage(Math.floor(x/2), Math.floor(y/2), 2*constants.TILEWIDTH, 2*constants.TILEHEIGHT);
    }
}

export default GhostSprites;