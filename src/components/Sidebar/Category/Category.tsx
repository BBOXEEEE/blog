import React from "react";

import { Link } from "gatsby";

import * as styles from "./Category.module.scss";

type Props = {
  category: Array<{
    label: string;
    path: string;
    count: number;
  }>;
};

const Category: React.FC<Props> = ({ category }: Props) => (
<>
  <div className={styles.title}>
    <Link to='/categories' className={styles.link}>
      Category
    </Link>
  </div>
  <nav className={styles.category}>
    <ul className={styles.list}>
      {category?.map((item) => (
        <li className={styles.item} key={item.path}>
          <Link
            to={item.path}
            className={styles.link}
            activeClassName={styles.active}
          >
            {item.label} ({item.count})
          </Link>
        </li>
      ))}
    </ul>
  </nav>
</>
);

export default Category;
